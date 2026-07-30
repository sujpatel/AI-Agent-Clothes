from dataclasses import dataclass

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


def get_conversation(user_id: str, token: str) -> ConversationSession | None:
    client = _client_for(token)
    result = client.table("conversations").select("contents, occasion").eq("user_id", user_id).execute()
    if not result.data:
        return None
    row = result.data[0]
    return ConversationSession(contents=row["contents"], occasion=row["occasion"])


def save_conversation(user_id: str, token: str, contents: list, occasion: str) -> None:
    client = _client_for(token)
    client.table("conversations").upsert({"user_id": user_id, "contents": contents, "occasion": occasion}).execute()
