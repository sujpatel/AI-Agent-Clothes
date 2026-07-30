import json
from pathlib import Path

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY
from app.gemini_utils import with_gemini_retry
from app.models import ClothingItem

# A hung network call would otherwise hang the whole request indefinitely.
client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=30_000))

PROMPT = """Look at this clothing item photo and extract its attributes.

- category: one of "top", "bottom", "shoes", "outerwear", "accessory"
- color: the dominant color
- formality: integer 1-5 (1 = very casual, 5 = very formal)
- warmth: integer 1-5 (1 = very light, 5 = very warm)
- pattern: e.g. "solid", "striped", "plaid", "graphic", "floral"
- description: one sentence describing the item, suitable for semantic search
  (e.g. "Navy blue merino wool crewneck sweater, casual-to-smart-casual, warm, solid color")
"""


@with_gemini_retry
def tag_photo(photo_path: Path) -> ClothingItem:
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
            response_schema=ClothingItem,
        ),
    )
    return ClothingItem.model_validate(json.loads(response.text))


def tag_folder(folder: Path) -> dict[str, ClothingItem]:
    results = {}
    for photo_path in sorted(folder.iterdir()):
        if photo_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        print(f"Tagging {photo_path.name}...")
        results[photo_path.name] = tag_photo(photo_path)
    return results
