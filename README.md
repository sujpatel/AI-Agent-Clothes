# RAG Wardrobe Assistant — Backend

FastAPI backend for an AI outfit assistant. Users photograph their clothes, the
backend detects and tags each garment, stores it in a vector database, and
recommends a daily outfit using the user's wardrobe and the live weather.

Frontend (React Native / Expo): [AI-Agent-Clothes-Frontend](https://github.com/sujpatel/AI-Agent-Clothes-Frontend)

## How it works

**Ingestion**
1. A user uploads one photo of several garments laid out (`POST /items/batch/detect`).
2. Gemini detects each garment and returns a bounding box per item. Malformed,
   inverted, and duplicate boxes (IoU ≥ 0.6) are dropped.
3. Each item is cropped and tagged by Gemini into 6 attributes: category, color,
   formality, warmth, pattern, and a one-line description. The detector's label
   is passed to the tagger as a hint, so crops that clip a neighboring garment
   are still tagged as the right item.
4. The user reviews the items in the app and confirms them
   (`POST /items/batch/confirm`). Each description is embedded with
   `sentence-transformers` and stored in Pinecone with its attributes.

**Outfit recommendation**
1. For each category, Pinecone returns the items closest to the occasion (e.g.
   "casual dinner"), filtered to the user and to items not in the laundry, and
   re-ranked so recently worn items aren't picked every day.
2. Gemini picks an outfit from those candidates in a multi-step tool-calling
   loop (capped at 5 steps). It can call a weather tool backed by
   [Open-Meteo](https://open-meteo.com/).
3. The conversation is saved, so users can follow up with corrections
   ("something warmer") through `POST /outfit/override`.

**Security and reliability**
- Supabase JWTs are verified against the project's JWKS (ES256) on every request.
- Row-Level Security on the `conversations` table and the `photos` storage
  bucket limits each user to their own data. Pinecone queries are filtered by
  `user_id`.
- Gemini calls retry with exponential backoff on 5xx errors and dropped connections.
- Endpoints are rate-limited with `slowapi`.

## Evaluation

`eval_flatlay.py` scores the detect → crop → tag pipeline against hand-labeled
flat-lay photos (108 labeled items across 10 photos). Predicted items are matched
to labeled items by category, and each run is saved to `eval/results/`.

| Version | Precision | Recall | F1 |
|---|---|---|---|
| Baseline | 69% | 96%* | 81% |
| + duplicate-box removal | 86% | 87% | 87% |
| + detector-label hint for tagging | **95%** (avg. of 2 runs) | **88%** | **91%** |

\*Inflated: the baseline returned 162 boxes for 108 items, so labels were often
matched by chance.

The most common remaining misses are small accessories (bracelets, belts) and
pairs of shoes boxed as two separate items.

```bash
python eval_flatlay.py                 # detect + tag every labeled photo
python eval_flatlay.py --detect-only   # item counts only, no tagging calls
python eval_flatlay.py --only test7.jpeg
```

The eval photos aren't in the repo, because most are third-party stock images.
To run the eval, add your own photos to `eval/photos/` and list their items in
`eval/labels.json`.

## API

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/items` | Upload and tag a single item |
| `GET` | `/items` | List the user's wardrobe |
| `POST` | `/items/batch/detect` | Detect and tag every garment in one photo (for review) |
| `POST` | `/items/batch/confirm` | Save reviewed items to the wardrobe |
| `DELETE` | `/items/pending/{item_id}` | Discard an unconfirmed item |
| `PATCH` | `/items/{item_id}` | Edit an item's attributes |
| `DELETE` | `/items/{item_id}` | Delete an item |
| `GET` | `/photos/{item_id}` | Fetch an item's photo |
| `GET` | `/outfit/today` | Recommend an outfit for an occasion and location |
| `POST` | `/outfit/override` | Refine the last outfit with a correction |
| `POST` | `/laundry/add` | Mark items unavailable |
| `POST` | `/laundry/finish` | Mark items available again |
| `GET` | `/health` | Health check |

All routes except `/health` require a Supabase access token
(`Authorization: Bearer <token>`).

## Running locally

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=...
PINECONE_API_KEY=...
SUPABASE_API_URL=https://<project>.supabase.co
SUPABASE_JWKS_URL=https://<project>.supabase.co/auth/v1/.well-known/jwks.json
SUPABASE_PUBLISHABLE_API_KEY=...
```

Run the `supabase_*.sql` files in the Supabase SQL editor to create the
`conversations` table and the storage policies.

## Deployment

Deployed on [Railway](https://railway.app/) using the `Procfile`.
