from pinecone import Pinecone, ServerlessSpec

from app.config import PINECONE_API_KEY
from app.embeddings import embed_text
from app.models import ClothingItem

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


def add_item(item_id: str, item: ClothingItem, photo_path: str) -> None:
    _index.upsert(
        vectors=[
            {
                "id": item_id,
                "values": embed_text(item.description),
                "metadata": {
                    "category": item.category,
                    "color": item.color,
                    "formality": item.formality,
                    "warmth": item.warmth,
                    "pattern": item.pattern,
                    "description": item.description,
                    "photo_path": photo_path,
                },
            }
        ]
    )


def query_candidates(occasion: str, category: str, n_results: int = 5) -> list[dict]:
    results = _index.query(
        vector=embed_text(occasion),
        top_k=n_results,
        filter={"category": {"$eq": category}},
        include_metadata=True,
    )

    candidates = []
    for match in results["matches"]:
        candidates.append({"id": match["id"], **match["metadata"]})
    return candidates
