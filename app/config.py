import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
SUPABASE_API_URL = os.environ["SUPABASE_API_URL"]
SUPABASE_JWKS_URL = os.environ["SUPABASE_JWKS_URL"]
SUPABASE_PUBLISHABLE_API_KEY = os.environ["SUPABASE_PUBLISHABLE_API_KEY"]
