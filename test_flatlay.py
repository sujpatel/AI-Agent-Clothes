"""Throwaway test: can Gemini reliably detect individual garments in one
'dumped on the bed' flat-lay photo?

Usage:
    python test_flatlay.py path/to/flatlay.jpg

It asks Gemini for a bounding box per clothing item, prints what it found,
and writes an annotated copy (<name>_annotated.jpg) with the boxes drawn on
so you can eyeball the accuracy before we build any UI around this.

Needs Pillow for the drawing step:  pip install pillow
"""

import json
import sys
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

from app.ingestion import client


class Detection(BaseModel):
    label: str          # what Gemini thinks the garment is, e.g. "white t-shirt"
    box_2d: list[int]   # [ymin, xmin, ymax, xmax], normalized 0-1000 (Gemini's convention)


class Detections(BaseModel):
    items: list[Detection]


PROMPT = """Detect every individual clothing item in this image (each top, bottom,
shoe, jacket, accessory, etc. — treat a pair of shoes as one item).

For each garment return:
- label: a short description (e.g. "blue jeans", "white t-shirt")
- box_2d: its bounding box as [ymin, xmin, ymax, xmax], normalized to 0-1000.

Only include actual clothing/footwear/accessories. Ignore the background,
hangers, and surfaces.
"""


def detect(photo_path: Path) -> Detections:
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

    # Gemini doesn't always respect the 4-number box format on cluttered
    # images — drop any malformed boxes instead of crashing the whole run.
    valid_items = []
    for item in raw.get("items", []):
        box = item.get("box_2d", [])
        if len(box) == 4:
            valid_items.append(item)
        else:
            print(f"  (skipping malformed detection: {item.get('label')!r} had box_2d={box})")

    return Detections.model_validate({"items": valid_items})


def annotate(photo_path: Path, detections: Detections) -> Path:
    """Draw the boxes + labels onto a copy of the image so accuracy is visible."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("\n(Pillow not installed — skipping the annotated image.")
        print(" Run 'pip install pillow' to see the boxes drawn on the photo.)")
        return photo_path

    image = Image.open(photo_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    try:
        font = ImageFont.truetype("arial.ttf", max(14, width // 60))
    except OSError:
        font = ImageFont.load_default()

    try:
        number_font = ImageFont.truetype("arial.ttf", max(20, width // 35))
    except OSError:
        number_font = font

    for i, det in enumerate(detections.items, 1):
        ymin, xmin, ymax, xmax = det.box_2d
        # Gemini boxes are normalized to 0-1000 — scale back to pixels.
        left = xmin / 1000 * width
        top = ymin / 1000 * height
        right = xmax / 1000 * width
        bottom = ymax / 1000 * height

        draw.rectangle([left, top, right, bottom], outline=(220, 40, 40), width=4)

        # Big number in the corner (readable even when boxes overlap) — cross-reference
        # it against the numbered list printed in the terminal to see the actual label.
        label_text = str(i)
        text_bbox = draw.textbbox((0, 0), label_text, font=number_font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([left, top, left + text_w + 10, top + text_h + 10], fill=(220, 40, 40))
        draw.text((left + 5, top + 3), label_text, fill=(255, 255, 255), font=number_font)

    out_path = photo_path.with_name(f"{photo_path.stem}_annotated.jpg")
    image.save(out_path)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_flatlay.py path/to/flatlay.jpg")
        sys.exit(1)

    photo_path = Path(sys.argv[1])
    if not photo_path.exists():
        print(f"No such file: {photo_path}")
        sys.exit(1)

    print(f"Detecting garments in {photo_path.name}...\n")
    detections = detect(photo_path)

    print(f"Gemini found {len(detections.items)} item(s):")
    for i, det in enumerate(detections.items, 1):
        print(f"  {i}. {det.label:<28} box(ymin,xmin,ymax,xmax)={det.box_2d}")

    out_path = annotate(photo_path, detections)
    if out_path != photo_path:
        print(f"\nAnnotated image saved to: {out_path}")
        print("Open it and check: is there one tight box per garment, nothing missed, nothing doubled?")
