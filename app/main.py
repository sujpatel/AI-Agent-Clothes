import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from app.composition import compose_outfit, override_outfit
from app.detection import crop_detections, detect_garments
from app.ingestion import tag_photo
from app.models import ClothingItem, Category
from app.uploads import read_validated_image, save_validated_image
from app.vectorstore import add_item, delete_item, list_items, set_availability, update_item_fields

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()


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

PHOTOS_DIR = Path(__file__).parent.parent / "data" / "photos"
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

# Serves every file in data/photos/ over HTTP, e.g. GET /photos/test1.jpg
app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")


def resolve_photo_path(item_id: str) -> Path:
    """Turns a client-supplied item_id into a path guaranteed to stay inside
    PHOTOS_DIR — item_id is untrusted input, and a naive PHOTOS_DIR / item_id
    join would let something like "../../etc/passwd" escape the folder."""
    photo_path = (PHOTOS_DIR / item_id).resolve()
    if not photo_path.is_relative_to(PHOTOS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid item id.")
    return photo_path

# Single-user prototype: remember the last conversation (and its occasion, so
# overrides can always re-check current availability) in memory.
# Replace with per-user session storage once auth/multi-user is added.
_last_conversation: list | None = None
_last_occasion: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/items")
async def create_item(photo: UploadFile):
    data = await read_validated_image(photo)

    item_id = f"{uuid.uuid4()}.jpg"
    photo_path = PHOTOS_DIR / item_id
    save_validated_image(photo_path, data)

    item = tag_photo(photo_path)
    add_item(item_id=item_id, item=item, photo_path=str(photo_path))

    return {"item_id": item_id, **item.model_dump()}


@app.get("/items")
def get_items():
    return list_items()


@app.post("/items/batch/detect")
async def detect_batch(photo: UploadFile):
    """Accept one flat-lay photo of multiple garments, detect + crop each
    item, and tag them individually. Items are saved as real photo files
    but NOT yet added to Pinecone — the client reviews them first and
    calls /items/batch/confirm to actually save the ones it wants to keep."""
    data = await read_validated_image(photo)

    dump_path = PHOTOS_DIR / f"dump-{uuid.uuid4()}.jpg"
    save_validated_image(dump_path, data)

    try:
        detections = detect_garments(dump_path)
        crop_paths = crop_detections(dump_path, detections, output_dir=PHOTOS_DIR)
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
def confirm_batch(request: BatchConfirmRequest):
    """Save the reviewed/edited items from a batch detection into Pinecone."""
    added = []
    for entry in request.items:
        photo_path = resolve_photo_path(entry.item_id)
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
        add_item(item_id=entry.item_id, item=item, photo_path=str(photo_path))
        added.append(entry.item_id)

    return {"added": added}


@app.delete("/items/pending/{item_id}")
def discard_pending_item(item_id: str):
    """Discard a crop the user rejected during batch review, before it's
    ever added to Pinecone (delete_item won't work here — it looks up the
    photo path via Pinecone metadata, which doesn't exist yet for these)."""
    resolve_photo_path(item_id).unlink(missing_ok=True)
    return {"discarded": item_id}


class UpdateItemRequest(BaseModel):
    category: Category | None = None
    color: str | None = None
    pattern: str | None = None
    formality: int | None = Field(default=None, ge=1, le=5)
    warmth: int | None = Field(default=None, ge=1, le=5)


@app.patch("/items/{item_id}")
def update_item(item_id: str, request: UpdateItemRequest):
    fields = {k: v for k, v in request.model_dump().items() if v is not None}
    if fields:
        update_item_fields(item_id, fields)
    return {"item_id": item_id, "updated": fields}


@app.delete("/items/{item_id}")
def remove_item(item_id: str):
    delete_item(item_id)
    return {"deleted": item_id}


@app.get("/outfit/today")
def get_todays_outfit(occasion: str = "casual", location: str = "Chicago", style_profile: str | None = None):
    global _last_conversation, _last_occasion
    contents, outfit = compose_outfit(occasion=occasion, location=location, style_profile=style_profile)
    _last_conversation = contents
    _last_occasion = occasion
    return outfit


class OverrideRequest(BaseModel):
    correction: str
    new_occasion: str | None = None


@app.post("/outfit/override")
def override(request: OverrideRequest):
    global _last_conversation, _last_occasion
    if _last_conversation is None or _last_occasion is None:
        # Real error, not a valid outfit shape — must be a 4xx so callers can't
        # mistake this for a successful response (e.g. after a server restart
        # wipes this in-memory conversation state).
        raise HTTPException(
            status_code=409, detail="No outfit has been generated yet. Call GET /outfit/today first."
        )

    contents, outfit = override_outfit(
        contents=_last_conversation,
        correction=request.correction,
        occasion=_last_occasion,
        new_occasion=request.new_occasion,
    )
    _last_conversation = contents
    if request.new_occasion:
        _last_occasion = request.new_occasion
    return outfit


class LaundryRequest(BaseModel):
    item_ids: list[str]


@app.post("/laundry/add")
def add_to_laundry(request: LaundryRequest):
    for item_id in request.item_ids:
        set_availability(item_id, available=False)
    return {"added": request.item_ids}


@app.post("/laundry/finish")
def finish_laundry(request: LaundryRequest):
    for item_id in request.item_ids:
        set_availability(item_id, available=True)
    return {"cleaned": request.item_ids}
