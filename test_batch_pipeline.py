"""Throwaway test: run the real detect -> crop -> tag pipeline (app/detection.py
+ app/ingestion.py) end to end on a flat-lay photo, exactly like the new
/items/batch/detect endpoint does, without needing the server running.

Usage:
    python test_batch_pipeline.py path/to/flatlay.jpg
"""

import sys
from pathlib import Path

from app.detection import crop_detections, detect_garments
from app.ingestion import tag_photo

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_batch_pipeline.py path/to/flatlay.jpg")
        sys.exit(1)

    photo_path = Path(sys.argv[1])
    if not photo_path.exists():
        print(f"No such file: {photo_path}")
        sys.exit(1)

    output_dir = Path(__file__).parent / "test_batch_output"

    print(f"Detecting garments in {photo_path.name}...")
    detections = detect_garments(photo_path)
    print(f"Found {len(detections.items)} item(s), cropping...")

    crop_paths = crop_detections(photo_path, detections, output_dir=output_dir)

    print(f"\nTagging each crop (this calls Gemini once per item — may take a bit):\n")
    for det, crop_path in zip(detections.items, crop_paths):
        item = tag_photo(crop_path)
        print(f"  {crop_path.name}")
        print(f"    detected label: {det.label}")
        print(f"    tagged as:      category={item.category}, color={item.color}, "
              f"pattern={item.pattern}, formality={item.formality}, warmth={item.warmth}")
        print(f"    description:    {item.description}\n")

    print(f"All crops saved in: {output_dir}")
    print("Open that folder and check: does each crop look like a clean, usable photo of one item?")
