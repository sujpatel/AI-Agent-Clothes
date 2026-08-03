import logging
from pathlib import Path

import jwt
from httpx import HTTPStatusError
from supabase import Client, create_client

from app.config import SUPABASE_API_URL, SUPABASE_PUBLISHABLE_API_KEY

logger = logging.getLogger(__name__)

BUCKET = "photos"


def _client_for(token: str) -> Client:
    """A Supabase client acting as the given user, so the storage bucket's
    RLS policies scope every request to that user's own folder."""
    client = create_client(SUPABASE_API_URL, SUPABASE_PUBLISHABLE_API_KEY)
    client.postgrest.auth(token)
    client.storage.session.headers["authorization"] = f"Bearer {token}"
    return client


def storage_path(user_id: str, item_id: str) -> str:
    return f"{user_id}/{item_id}"


def upload_photo(user_id: str, token: str, item_id: str, local_path: Path) -> None:
    """Uploads a local file (already validated/re-encoded by uploads.py) to
    this user's folder in the photos bucket — the permanent, durable home
    for the photo. The local file is disposable after this succeeds."""
    # Temporary — decode the token's own claim of "who am I" and compare it
    # directly to the folder path being written, to rule in/out a mismatch
    # as the cause of the RLS rejection.
    try:
        unverified = jwt.decode(token, options={"verify_signature": False})
        logger.error(f"upload_photo: token sub={unverified.get('sub')!r} path={storage_path(user_id, item_id)!r}")
    except Exception as e:
        logger.error(f"upload_photo: could not decode token for comparison: {e}")

    client = _client_for(token)
    with open(local_path, "rb") as f:
        try:
            client.storage.from_(BUCKET).upload(
                storage_path(user_id, item_id), f, file_options={"content-type": "image/jpeg", "upsert": "true"}
            )
        except HTTPStatusError as e:
            # Temporary — logs Supabase's actual error body instead of just
            # the bare status code, to diagnose a live upload failure.
            logger.error(f"Storage upload failed: {e.response.status_code} {e.response.text}")
            raise


def download_photo(user_id: str, token: str, item_id: str) -> bytes | None:
    """Fetches a photo's bytes from storage. Returns None if it doesn't
    exist or doesn't belong to this user (RLS blocks the latter silently)."""
    client = _client_for(token)
    try:
        return client.storage.from_(BUCKET).download(storage_path(user_id, item_id))
    except Exception:
        return None


def delete_photo(user_id: str, token: str, item_id: str) -> None:
    client = _client_for(token)
    client.storage.from_(BUCKET).remove([storage_path(user_id, item_id)])
