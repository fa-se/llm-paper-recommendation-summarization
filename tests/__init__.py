import os

# core/__init__.py creates an OpenAI client at import time, which needs a key; the tests make no API calls
os.environ.setdefault("OPENAI_API_KEY", "unused")
