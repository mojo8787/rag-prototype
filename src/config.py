"""Configuration for chunking, retrieval, and human-review gates."""
import os
from dotenv import load_dotenv

load_dotenv()

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Retrieval
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# Human-review gates
GATE_CONFIDENCE_THRESHOLD = float(os.getenv("GATE_CONFIDENCE_THRESHOLD", "0.7"))
GATE_MIN_CHUNKS = int(os.getenv("GATE_MIN_CHUNKS", "1"))

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Azure OpenAI (when LLM_PROVIDER=azure)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
