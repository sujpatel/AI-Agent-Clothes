import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.composition import compose_outfit, override_outfit
from app.ingestion import tag_photo
from app.models import Category
from app.vectorstore import add_item, delete_item, list_items, set_availability, update_item_fields

app = FastAPI()

PHOTOS_DIR = Path(__file__).parent.parent / "data" / "photos"
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

# Serves every file in data/photos/ over HTTP, e.g. GET /photos/test1.jpg
app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

# Single-user prototype: remember the last conversation in memory so overrides can continue it.
# Replace with per-user session storage once auth/multi-user is added.
_last_conversation: list | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/items")
async def create_item(photo: UploadFile):
    item_id = f"{uuid.uuid4()}{Path(photo.filename).suffix}"
    photo_path = PHOTOS_DIR / item_id

    photo_path.write_bytes(await photo.read())

    item = tag_photo(photo_path)
    add_item(item_id=item_id, item=item, photo_path=str(photo_path))

    return {"item_id": item_id, **item.model_dump()}


@app.get("/items")
def get_items():
    return list_items()


class UpdateItemRequest(BaseModel):
    category: Category | None = None
    color: str | None = None
    pattern: str | None = None
    formality: int | None = None
    warmth: int | None = None


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
def get_todays_outfit(occasion: str = "casual", location: str = "Chicago"):
    global _last_conversation
    contents, outfit = compose_outfit(occasion=occasion, location=location)
    _last_conversation = contents
    return outfit


class OverrideRequest(BaseModel):
    correction: str
    new_occasion: str | None = None


@app.post("/outfit/override")
def override(request: OverrideRequest):
    global _last_conversation
    if _last_conversation is None:
        return {"error": "No outfit has been generated yet. Call GET /outfit/today first."}

    contents, outfit = override_outfit(
        contents=_last_conversation,
        correction=request.correction,
        new_occasion=request.new_occasion,
    )
    _last_conversation = contents
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
