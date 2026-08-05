import logging
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi import Response
from fastapi.responses import FileResponse, JSONResponse
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.auth import AuthedUser, get_current_user, get_current_user_id
from app.composition import compose_outfit, override_outfit
from app.detection import crop_detections, detect_garments
from app.ingestion import tag_photo
from app.models import ClothingItem, Category
from app.photo_store import download_photo, upload_photo
from app.session_store import get_conversation, save_conversation
from app.uploads import read_validated_image, save_validated_image
from app.vectorstore import add_item, delete_item, list_items, set_availability, update_item_fields

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.error("=== BUILD MARKER deploy-check-9f3k2 ===")


def _rate_limit_key(request: Request) -> str:
    """Key by the authenticated user's id when available, so limits track a
    real identity rather than an IP address (unreliable behind Railway's
    proxy, and shared by anyone on the same network/NAT). Falls back to IP
    for unauthenticated requests (just /health)."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            from app.auth import _verify

            return _verify(auth_header).user_id
        except HTTPException:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(genai_errors.ServerError)
async def gemini_server_error_handler(request: Request, exc: genai_errors.ServerError):
    # Reached when Gemini is still overloaded/unavailable after the retries in
    # gemini_utils.with_gemini_retry are exhausted — return a clean error
    # instead of a raw 500 traceback.
    logger.error(f"Gemini unavailable for {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=503,
        content={"error": "The AI service is temporarily overloaded. Please try again in a moment."},
    )


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError):
    # Raised by vectorstore._check_ownership when an item doesn't belong to
    # the requesting user — without this handler it falls through to a raw
    # 500 instead of a clean 403.
    return JSONResponse(status_code=403, content={"error": str(exc)})

PHOTOS_DIR = Path(__file__).parent.parent / "data" / "photos"
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_photo_path(user_id: str, item_id: str) -> Path:
    """Turns a client-supplied item_id into a path guaranteed to stay inside
    that user's own photo folder — item_id is untrusted input, and a naive
    join would let something like "../../etc/passwd" escape the folder, or
    let one user read another user's path by guessing their item_id."""
    user_dir = (PHOTOS_DIR / user_id).resolve()
    photo_path = (user_dir / item_id).resolve()
    if not photo_path.is_relative_to(user_dir):
        raise HTTPException(status_code=400, detail="Invalid item id.")
    return photo_path


@app.get("/photos/{item_id}")
def get_photo(item_id: str, user: AuthedUser = Depends(get_current_user)):
    """Serves a photo only to the user who owns it. Confirmed items live in
    Supabase Storage; unconfirmed batch-review crops still live on local
    disk until the user confirms or discards them, so check both."""
    data = download_photo(user.user_id, user.token, item_id)
    if data is not None:
        return Response(content=data, media_type="image/jpeg")

    photo_path = resolve_photo_path(user.user_id, item_id)
    if not photo_path.is_file():
        raise HTTPException(status_code=404, detail="Photo not found.")
    return FileResponse(photo_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/items")
@limiter.limit("20/minute")
async def create_item(request: Request, photo: UploadFile, user: AuthedUser = Depends(get_current_user)):
    data = await read_validated_image(photo)

    item_id = f"{uuid.uuid4()}.jpg"
    user_dir = PHOTOS_DIR / user.user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    temp_path = user_dir / item_id
    save_validated_image(temp_path, data)

    try:
        item = tag_photo(temp_path)
        upload_photo(user.user_id, user.token, item_id, temp_path)
        add_item(item_id=item_id, item=item, user_id=user.user_id)
    finally:
        # The local file was only ever a working copy for Pillow/Gemini to
        # read — the durable copy now lives in Supabase Storage.
        temp_path.unlink(missing_ok=True)

    return {"item_id": item_id, **item.model_dump()}


@app.get("/items")
def get_items(user_id: str = Depends(get_current_user_id)):
    return list_items(user_id=user_id)


@app.post("/items/batch/detect")
@limiter.limit("10/minute")
async def detect_batch(request: Request, photo: UploadFile, user_id: str = Depends(get_current_user_id)):
    """Accept one flat-lay photo of multiple garments, detect + crop each
    item, and tag them individually. Items are saved as real photo files
    but NOT yet added to Pinecone — the client reviews them first and
    calls /items/batch/confirm to actually save the ones it wants to keep."""
    data = await read_validated_image(photo)

    user_dir = PHOTOS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    dump_path = user_dir / f"dump-{uuid.uuid4()}.jpg"
    save_validated_image(dump_path, data)

    try:
        detections = detect_garments(dump_path)
        crop_paths = crop_detections(dump_path, detections, output_dir=user_dir)
    finally:
        dump_path.unlink(missing_ok=True)

    pending = []
    for crop_path in crop_paths:
        try:
            item = tag_photo(crop_path)
        except Exception:
            # One crop failing to tag (even after retries) shouldn't waste the
            # whole batch — skip it and clean up its orphaned file.
            logger.warning(f"Skipping crop {crop_path.name} — tagging failed", exc_info=True)
            crop_path.unlink(missing_ok=True)
            continue
        pending.append({"item_id": crop_path.name, **item.model_dump()})

    return {"items": pending}


class BatchConfirmItem(BaseModel):
    item_id: str
    category: Category
    color: str
    formality: int = Field(ge=1, le=5)
    warmth: int = Field(ge=1, le=5)
    pattern: str
    description: str


class BatchConfirmRequest(BaseModel):
    items: list[BatchConfirmItem]


@app.post("/items/batch/confirm")
def confirm_batch(request: BatchConfirmRequest, user: AuthedUser = Depends(get_current_user)):
    """Save the reviewed/edited items from a batch detection into Pinecone,
    uploading each confirmed crop to Supabase Storage as its durable copy."""
    added = []
    for entry in request.items:
        photo_path = resolve_photo_path(user.user_id, entry.item_id)
        if not photo_path.is_file():
            raise HTTPException(status_code=404, detail=f"No pending photo found for '{entry.item_id}'.")

        item = ClothingItem(
            category=entry.category,
            color=entry.color,
            formality=entry.formality,
            warmth=entry.warmth,
            pattern=entry.pattern,
            description=entry.description,
        )
        upload_photo(user.user_id, user.token, entry.item_id, photo_path)
        add_item(item_id=entry.item_id, item=item, user_id=user.user_id)
        photo_path.unlink(missing_ok=True)
        added.append(entry.item_id)

    return {"added": added}


@app.delete("/items/pending/{item_id}")
def discard_pending_item(item_id: str, user_id: str = Depends(get_current_user_id)):
    """Discard a crop the user rejected during batch review — these crops
    are still local-only (never uploaded to storage) until confirmed, so a
    plain local file delete is all that's needed."""
    resolve_photo_path(user_id, item_id).unlink(missing_ok=True)
    return {"discarded": item_id}


class UpdateItemRequest(BaseModel):
    category: Category | None = None
    color: str | None = None
    pattern: str | None = None
    formality: int | None = Field(default=None, ge=1, le=5)
    warmth: int | None = Field(default=None, ge=1, le=5)


@app.patch("/items/{item_id}")
def update_item(item_id: str, request: UpdateItemRequest, user_id: str = Depends(get_current_user_id)):
    fields = {k: v for k, v in request.model_dump().items() if v is not None}
    if fields:
        update_item_fields(item_id, fields, user_id=user_id)
    return {"item_id": item_id, "updated": fields}


@app.delete("/items/{item_id}")
def remove_item(item_id: str, user: AuthedUser = Depends(get_current_user)):
    delete_item(item_id, user_id=user.user_id, token=user.token)
    return {"deleted": item_id}


@app.get("/outfit/today")
@limiter.limit("15/minute")
def get_todays_outfit(
    request: Request,
    occasion: str = "casual",
    location: str = "Chicago",
    style_profile: str | None = None,
    user: AuthedUser = Depends(get_current_user),
):
    contents, outfit = compose_outfit(
        occasion=occasion, location=location, user_id=user.user_id, style_profile=style_profile
    )
    save_conversation(user.user_id, user.token, contents, occasion)
    return outfit


class OverrideRequest(BaseModel):
    correction: str
    new_occasion: str | None = None


@app.post("/outfit/override")
@limiter.limit("20/minute")
def override(request: Request, body: OverrideRequest, user: AuthedUser = Depends(get_current_user)):
    session = get_conversation(user.user_id, user.token)
    if session is None:
        # Real error, not a valid outfit shape — must be a 4xx so callers can't
        # mistake this for a successful response (e.g. after a server restart
        # or if this user has never generated an outfit yet).
        raise HTTPException(
            status_code=409, detail="No outfit has been generated yet. Call GET /outfit/today first."
        )

    contents, outfit = override_outfit(
        contents=session.contents,
        correction=body.correction,
        occasion=session.occasion,
        user_id=user.user_id,
        new_occasion=body.new_occasion,
    )
    save_conversation(user.user_id, user.token, contents, body.new_occasion or session.occasion)
    return outfit


class LaundryRequest(BaseModel):
    item_ids: list[str]


@app.post("/laundry/add")
def add_to_laundry(request: LaundryRequest, user_id: str = Depends(get_current_user_id)):
    for item_id in request.item_ids:
        set_availability(item_id, available=False, user_id=user_id)
    return {"added": request.item_ids}


@app.post("/laundry/finish")
def finish_laundry(request: LaundryRequest, user_id: str = Depends(get_current_user_id)):
    for item_id in request.item_ids:
        set_availability(item_id, available=True, user_id=user_id)
    return {"cleaned": request.item_ids}
