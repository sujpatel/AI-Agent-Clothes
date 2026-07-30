from dataclasses import dataclass

import jwt
from fastapi import Header, HTTPException

from app.config import SUPABASE_JWKS_URL

# Caches the fetched JWKS and re-fetches automatically if a token references
# a kid it doesn't recognize yet (e.g. after Supabase rotates keys).
_jwks_client = jwt.PyJWKClient(SUPABASE_JWKS_URL)


@dataclass
class AuthedUser:
    user_id: str
    token: str  # the raw JWT, needed to make RLS-scoped Supabase requests


def _verify(authorization: str) -> AuthedUser:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization.removeprefix("Bearer ").strip()

    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(token, signing_key.key, algorithms=["ES256"], audience="authenticated")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    return AuthedUser(user_id=payload["sub"], token=token)


def get_current_user_id(authorization: str = Header(...)) -> str:
    """FastAPI dependency: verifies the Supabase-issued JWT on an incoming
    request and returns the authenticated user's id. Raises 401 if the token
    is missing, malformed, expired, or fails signature verification."""
    return _verify(authorization).user_id


def get_current_user(authorization: str = Header(...)) -> AuthedUser:
    """Same verification as get_current_user_id, but also returns the raw
    token — needed by routes that make RLS-scoped Supabase requests on the
    user's behalf (e.g. reading/writing their conversation session)."""
    return _verify(authorization)
