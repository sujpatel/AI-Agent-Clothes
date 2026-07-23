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
    bounding box per item. Drops any malformed boxes instead of raising,
    since Gemini doesn't always respect the 4-number format on cluttered images."""
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

    valid_items = [item for item in raw.get("items", []) if len(item.get("box_2d", [])) == 4]
    return Detections.model_validate({"items": valid_items})


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
