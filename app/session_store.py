from dataclasses import dataclass

from google.genai import types
from supabase import Client, create_client

from app.config import SUPABASE_API_URL, SUPABASE_PUBLISHABLE_API_KEY


@dataclass
class ConversationSession:
    contents: list
    occasion: str


def _client_for(token: str) -> Client:
    """A Supabase client acting as the given user, so Row Level Security
    scopes every request to that user's own conversation row."""
    client = create_client(SUPABASE_API_URL, SUPABASE_PUBLISHABLE_API_KEY)
    client.postgrest.auth(token)
    return client


def _serialize_contents(contents: list) -> list:
    """The Gemini conversation is a mix of plain strings and google-genai
    Content objects, which aren't JSON-serializable on their own — tag each
    entry so it can be restored to its original type on the way back out."""
    serialized = []
    for entry in contents:
        if isinstance(entry, types.Content):
            serialized.append({"type": "content", "value": entry.to_json_dict()})
        else:
            serialized.append({"type": "text", "value": entry})
    return serialized


def _deserialize_contents(serialized: list) -> list:
    contents = []
    for entry in serialized:
        if entry["type"] == "content":
            contents.append(types.Content.model_validate(entry["value"]))
        else:
            contents.append(entry["value"])
    return contents


def get_conversation(user_id: str, token: str) -> ConversationSession | None:
    client = _client_for(token)
    result = client.table("conversations").select("contents, occasion").eq("user_id", user_id).execute()
    if not result.data:
        return None
    row = result.data[0]
    return ConversationSession(contents=_deserialize_contents(row["contents"]), occasion=row["occasion"])


def save_conversation(user_id: str, token: str, contents: list, occasion: str) -> None:
    client = _client_for(token)
    client.table("conversations").upsert(
        {"user_id": user_id, "contents": _serialize_contents(contents), "occasion": occasion}
    ).execute()
