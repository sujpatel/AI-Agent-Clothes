import time

from pinecone import Pinecone, ServerlessSpec

from app.config import PINECONE_API_KEY
from app.embeddings import embed_text
from app.models import ClothingItem
from app.photo_store import delete_photo

INDEX_NAME = "wardrobe"
DIMENSION = 384  # matches all-MiniLM-L6-v2 output size

_pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in [index["name"] for index in _pc.list_indexes()]:
    _pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # required region on Starter plan
    )

_index = _pc.Index(INDEX_NAME)


def add_item(item_id: str, item: ClothingItem, user_id: str) -> None:
    # No photo_path stored — the photo's location in Supabase Storage is
    # always derivable as f"{user_id}/{item_id}", so there's nothing to keep
    # in sync separately.
    _index.upsert(
        vectors=[
            {
                "id": item_id,
                "values": embed_text(item.description),
                "metadata": {
                    "user_id": user_id,
                    "category": item.category,
                    "color": item.color,
                    "formality": item.formality,
                    "warmth": item.warmth,
                    "pattern": item.pattern,
                    "description": item.description,
                    "available": True,
                    "last_worn": 0,  # 0 = never worn; sorts as most eligible for rotation
                },
            }
        ]
    )


def _check_ownership(item_id: str, user_id: str) -> dict:
    """Fetches an item and raises if it doesn't belong to user_id — every
    mutation on a specific item_id must call this first, since Pinecone's
    update/delete-by-id have no built-in concept of ownership."""
    fetch_result = _index.fetch(ids=[item_id])
    vector = fetch_result.vectors.get(item_id)
    if vector is None or vector.metadata.get("user_id") != user_id:
        raise PermissionError(f"Item '{item_id}' not found for this user.")
    return vector.metadata


def set_availability(item_id: str, available: bool, user_id: str) -> None:
    _check_ownership(item_id, user_id)
    fields = {"available": available}
    if not available:
        # Sending something to laundry is the clearest signal it was actually
        # worn — reuses that existing action instead of needing a new one.
        fields["last_worn"] = int(time.time())
    _index.update(id=item_id, set_metadata=fields)


def update_item_fields(item_id: str, fields: dict, user_id: str) -> None:
    """Update specific metadata fields (e.g. category, color, pattern) without
    touching the embedding vector — fixes a mis-tagged item's fields directly."""
    _check_ownership(item_id, user_id)
    _index.update(id=item_id, set_metadata=fields)


def delete_item(item_id: str, user_id: str, token: str) -> None:
    _check_ownership(item_id, user_id)
    _index.delete(ids=[item_id])
    delete_photo(user_id, token, item_id)


def list_items(user_id: str) -> list[dict]:
    all_ids = []
    for batch in _index.list():
        all_ids.extend(item.id for item in batch.vectors)

    if not all_ids:
        return []

    fetch_result = _index.fetch(ids=all_ids)
    return [
        {"id": item_id, **vector.metadata}
        for item_id, vector in fetch_result.vectors.items()
        if vector.metadata.get("user_id") == user_id
    ]


POOL_SIZE_MULTIPLIER = 3  # pull a wider pool than needed so recency has room to reorder results
MAX_POOL_SIZE = 20
RECENCY_WINDOW_DAYS = 14  # items unworn for 2+ weeks get no rotation penalty at all
RECENCY_PENALTY_WEIGHT = 0.3  # tuned to reorder similarly-relevant items, not override a clearly better match


def query_candidates(occasion: str, category: str, user_id: str, n_results: int = 5) -> list[dict]:
    """Retrieves candidates by occasion relevance, then re-ranks a wider pool
    to deprioritize recently-worn items — so the same few favorites don't
    keep winning every time, without hard-excluding them like `available` does."""
    pool_size = min(n_results * POOL_SIZE_MULTIPLIER, MAX_POOL_SIZE)
    results = _index.query(
        vector=embed_text(occasion),
        top_k=pool_size,
        filter={"category": {"$eq": category}, "available": {"$eq": True}, "user_id": {"$eq": user_id}},
        include_metadata=True,
    )

    now = time.time()
    candidates = []
    for match in results["matches"]:
        metadata = match["metadata"]
        last_worn = metadata.get("last_worn", 0)
        days_since_worn = (now - last_worn) / 86400 if last_worn else RECENCY_WINDOW_DAYS

        recency_penalty = (
            max(0.0, RECENCY_WINDOW_DAYS - days_since_worn) / RECENCY_WINDOW_DAYS * RECENCY_PENALTY_WEIGHT
        )
        candidates.append({"id": match["id"], "_rank_score": match["score"] - recency_penalty, **metadata})

    candidates.sort(key=lambda c: c["_rank_score"], reverse=True)
    top_candidates = candidates[:n_results]
    for candidate in top_candidates:
        candidate.pop("_rank_score", None)

    return top_candidates
