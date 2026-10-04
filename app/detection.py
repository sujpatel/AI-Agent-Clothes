import json
import uuid
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

from app.gemini_utils import with_gemini_retry
from app.ingestion import client

PROMPT = """Detect every individual clothing item in this image (each top, bottom,
shoe, jacket, accessory, etc. — treat a pair of shoes as one item).

For each garment return:
- label: a short description (e.g. "blue jeans", "white t-shirt")
- box_2d: its bounding box as [ymin, xmin, ymax, xmax], normalized to 0-1000.

Only include actual clothing/footwear/accessories. Ignore the background,
hangers, and surfaces.
"""


class Detection(BaseModel):
    label: str
    box_2d: list[int]  # [ymin, xmin, ymax, xmax], normalized 0-1000


class Detections(BaseModel):
    items: list[Detection]


@with_gemini_retry
def detect_garments(photo_path: Path) -> Detections:
    """Ask Gemini to find every garment in a flat-lay photo and return a
    bounding box per item. Drops any malformed or inverted boxes instead of
    raising, since Gemini doesn't always return valid boxes on cluttered images."""
    image_bytes = photo_path.read_bytes()
    mime_type = "image/png" if photo_path.suffix.lower() == ".png" else "image/jpeg"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Detections,
        ),
    )
    raw = json.loads(response.text)

    valid_items = [item for item in raw.get("items", []) if _is_valid_box(item.get("box_2d", []))]
    return Detections.model_validate({"items": _drop_duplicate_boxes(valid_items)})


# Two boxes overlapping this much are treated as the same garment. Kept well
# above zero so items legitimately lying on each other (a belt on a dress)
# aren't merged — those have small overlap relative to their combined area.
DUPLICATE_IOU_THRESHOLD = 0.6


def _iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-union of two [ymin, xmin, ymax, xmax] boxes."""
    inter_h = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_w = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_h * inter_w
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _drop_duplicate_boxes(items: list[dict]) -> list[dict]:
    """Gemini sometimes returns the same garment twice with nearly identical
    boxes. Keep the larger box of any heavily-overlapping pair (it's the one
    less likely to have clipped the garment)."""
    by_size = sorted(
        items,
        key=lambda it: (it["box_2d"][2] - it["box_2d"][0]) * (it["box_2d"][3] - it["box_2d"][1]),
        reverse=True,
    )
    kept: list[dict] = []
    for item in by_size:
        if all(_iou(item["box_2d"], k["box_2d"]) < DUPLICATE_IOU_THRESHOLD for k in kept):
            kept.append(item)
    # Restore Gemini's original order so crops line up the way they did before.
    return [it for it in items if any(it is k for k in kept)]


def _is_valid_box(box: list[int]) -> bool:
    # Gemini occasionally returns inverted or zero-size boxes, which make
    # PIL's crop() raise, so they're dropped along with malformed ones.
    if len(box) != 4:
        return False
    ymin, xmin, ymax, xmax = box
    return ymax > ymin and xmax > xmin


def crop_detections(photo_path: Path, detections: Detections, output_dir: Path) -> list[Path]:
    """Crop each detected box out of the original photo and save it as its own
    file in output_dir, named with a fresh uuid. Returns the saved crop paths,
    in the same order as detections.items."""
    from PIL import Image

    image = Image.open(photo_path).convert("RGB")
    width, height = image.size

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = []

    for det in detections.items:
        ymin, xmin, ymax, xmax = det.box_2d
        left = xmin / 1000 * width
        top = ymin / 1000 * height
        right = xmax / 1000 * width
        bottom = ymax / 1000 * height

        crop = image.crop((left, top, right, bottom))
        crop_path = output_dir / f"{uuid.uuid4()}.jpg"
        crop.save(crop_path, "JPEG")
        crop_paths.append(crop_path)

    return crop_paths
