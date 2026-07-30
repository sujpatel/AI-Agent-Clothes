import json
from typing import Optional

from google.genai import types
from pydantic import BaseModel

from app.gemini_utils import with_gemini_retry
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


def _build_prompt(
    occasion: str, location: str, candidates_by_category: dict[str, list[dict]], style_profile: str | None = None
) -> str:
    style_guideline = (
        f"- The user's personal style is: {style_profile}. When multiple candidates are equally "
        "appropriate for the occasion, prefer the one that best fits this aesthetic — but occasion "
        "fit, formality, and weather always come first. Never sacrifice appropriateness for style.\n"
        if style_profile
        else ""
    )

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
{style_guideline}
Pick one item id per relevant category.

Then write your reasoning as ONE short, plain sentence (under 20 words) about the clothes and
conditions. Use location only as weather context — never name the location or phrase it like an
event (e.g. don't say "this outfit for {location}" or treat {location} as the occasion).
"""


MAX_REASONING_STEPS = 5


@with_gemini_retry
def _generate_reasoning_step(contents: list):
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=[WEATHER_TOOL]),
    )


@with_gemini_retry
def _generate_forced_final_step(contents: list):
    # No tools passed — Gemini can't call get_weather again, so it must answer.
    return client.models.generate_content(model="gemini-2.5-flash", contents=contents)


def _reason(contents: list) -> list:
    """Let Gemini reason over the conversation so far, calling the weather tool if it wants to.
    Returns the updated contents once Gemini has settled on a final answer (no more tool calls),
    or once MAX_REASONING_STEPS is hit (forces a final answer instead of looping forever)."""
    for _ in range(MAX_REASONING_STEPS):
        response = _generate_reasoning_step(contents)
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

    # Hit the step cap without a final answer — ask once more with no tools
    # available, so it's forced to answer instead of calling get_weather again.
    contents.append("Stop calling tools and give your best answer now based on what you already know.")
    response = _generate_forced_final_step(contents)
    contents.append(response.candidates[0].content)
    return contents


@with_gemini_retry
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


def compose_outfit(
    occasion: str, location: str, user_id: str, style_profile: str | None = None
) -> tuple[list, OutfitChoice]:
    candidates_by_category = {
        category: query_candidates(occasion, category=category, user_id=user_id, n_results=5)
        for category in CATEGORIES
    }

    contents = [_build_prompt(occasion, location, candidates_by_category, style_profile)]
    contents = _reason(contents)
    outfit = _finalize(contents)
    return contents, outfit


def override_outfit(
    contents: list,
    correction: str,
    occasion: str,
    user_id: str,
    new_occasion: Optional[str] = None,
) -> tuple[list, OutfitChoice]:
    """Apply a follow-up correction (e.g. "swap the shoes", "show me another",
    "it's a dinner tonight instead") to an existing conversation.

    Always re-retrieves candidates first, using the new occasion if it changed
    or the current one otherwise — wardrobe availability (e.g. something sent
    to laundry) may have changed since this conversation started, and without
    a fresh lookup every time, a correction could recommend an item that's no
    longer actually available."""
    effective_occasion = new_occasion or occasion
    candidates_by_category = {
        category: query_candidates(effective_occasion, category=category, user_id=user_id, n_results=5)
        for category in CATEGORIES
    }
    occasion_note = f"The occasion has changed to: {new_occasion}. " if new_occasion else ""
    contents.append(
        f"{occasion_note}Current available candidate items (availability may have changed):\n"
        f"{_format_candidates(candidates_by_category)}"
    )

    contents.append(correction)
    contents = _reason(contents)
    outfit = _finalize(contents)
    return contents, outfit
