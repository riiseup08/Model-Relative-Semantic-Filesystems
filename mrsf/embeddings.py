import numpy as np
import requests

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

def embed(text: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": text}
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"][0], dtype="float32")