import json
from typing import Optional

from google.genai import types
from pydantic import BaseModel

from app.ingestion import client
from app.vectorstore import query_candidates
from app.weather import get_weather

CATEGORIES = ["top", "bottom", "shoes", "outerwear"]

WEATHER_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get current weather conditions (temperature, precipitation) for a location.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        )
    ]
)


class OutfitChoice(BaseModel):
    top_id: Optional[str] = None
    bottom_id: Optional[str] = None
    shoes_id: Optional[str] = None
    outerwear_id: Optional[str] = None
    reasoning: str


def _format_candidates(candidates_by_category: dict[str, list[dict]]) -> str:
    lines = []
    for category, items in candidates_by_category.items():
        lines.append(f"\n{category.upper()} options:")
        if not items:
            lines.append("- (none available)")
            continue
        for item in items:
            lines.append(
                f"- id={item['id']}: {item['description']} "
                f"(formality={item['formality']}, warmth={item['warmth']})"
            )
    return "\n".join(lines)


def _build_prompt(occasion: str, location: str, candidates_by_category: dict[str, list[dict]]) -> str:
    return f"""You are choosing an outfit for the user.

Occasion: {occasion}
Location: {location}

Candidate items:
{_format_candidates(candidates_by_category)}

Guidelines:
- Call get_weather for the location if current conditions would affect your choice.
- Warmth ratings matter most for outerwear and tops in cold weather. Footwear should be judged
  on overall appropriateness (e.g. not sandals in snow), not primarily on its warmth rating.
- Formality should roughly match the occasion.
- Colors and patterns should be coordinated for a coherent look.
- Only choose outerwear if the weather or occasion calls for it; otherwise leave it out.

Pick one item id per relevant category and explain your reasoning in 1-2 sentences.
"""


def _reason(contents: list) -> list:
    """Let Gemini reason over the conversation so far, calling the weather tool if it wants to.
    Returns the updated contents once Gemini has settled on a final answer (no more tool calls)."""
    while True:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(tools=[WEATHER_TOOL]),
        )
        part = response.candidates[0].content.parts[0]

        if part.function_call:
            call = part.function_call
            result = get_weather(**call.args)
            contents.append(response.candidates[0].content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(name=call.name, response=result)],
                )
            )
            continue

        contents.append(response.candidates[0].content)
        return contents


def _finalize(contents: list) -> OutfitChoice:
    """Ask for the final structured choice, now that reasoning is settled."""
    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents + ["Now output your final chosen item ids and reasoning in the required JSON format."],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OutfitChoice,
        ),
    )
    return OutfitChoice.model_validate(json.loads(final_response.text))


def compose_outfit(occasion: str, location: str) -> tuple[list, OutfitChoice]:
    candidates_by_category = {
        category: query_candidates(occasion, category=category, n_results=5)
        for category in CATEGORIES
    }

    contents = [_build_prompt(occasion, location, candidates_by_category)]
    contents = _reason(contents)
    outfit = _finalize(contents)
    return contents, outfit


def override_outfit(
    contents: list,
    correction: str,
    new_occasion: Optional[str] = None,
) -> tuple[list, OutfitChoice]:
    """Apply a follow-up correction (e.g. "swap the shoes", "show me another",
    "it's a dinner tonight instead") to an existing conversation, re-retrieving
    candidates first if the occasion changed."""
    if new_occasion:
        candidates_by_category = {
            category: query_candidates(new_occasion, category=category, n_results=5)
            for category in CATEGORIES
        }
        contents.append(
            f"The occasion has changed to: {new_occasion}. Updated candidate items:\n"
            f"{_format_candidates(candidates_by_category)}"
        )

    contents.append(correction)
    contents = _reason(contents)
    outfit = _finalize(contents)
    return contents, outfit
