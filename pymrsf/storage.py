"""
pymrsf.storage — Delta compression storage + FAISS semantic search

Core functionality:
  - mrsf_write : Store text with delta compression (only "surprise" tokens)
  - mrsf_read  : Reconstruct text via KV-cached O(n) model inference
  - save_index : Persist FAISS index to disk
  - load_index : Load persisted FAISS index
"""

import sqlite3, msgpack, uuid, json, os
import numpy as np
import faiss

from .core import tokenize, detokenize, compute_delta, ModelSession, MODEL_VERSION
from .embeddings import embed

DB_PATH    = "mrsf.db"
FAISS_PATH = "mrsf.faiss"
EMBED_DIM  = 768

# Lazy initialization — not loaded at import time
_faiss_index = None
_index_meta  = None
_conn        = None
_cur         = None


def _get_db():
    """Lazy SQLite connection."""
    global _conn, _cur
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH)
        _cur  = _conn.cursor()
        _cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id         TEXT PRIMARY KEY,
            model_version  TEXT,
            delta          BLOB,
            token_count    INTEGER,
            surprise_count INTEGER
        )""")
        _conn.commit()
    return _cur, _conn


def _get_index():
    """Lazy FAISS index."""
    global _faiss_index, _index_meta
    if _faiss_index is None:
        _faiss_index = faiss.IndexHNSWFlat(EMBED_DIM, 32)
        _index_meta  = []
    return _faiss_index, _index_meta


def mrsf_write(text: str, doc_id: str = None) -> dict:
    """Store a document with delta compression.

    Args:
        text  : The text to store
        doc_id: Optional custom document ID (auto-generated if None)

    Returns:
        dict with doc_id, token_count, surprise_count, compression
    """
    doc_id    = doc_id or str(uuid.uuid4())
    token_ids = tokenize(text)
    n         = len(token_ids)

    # Compute delta (surprise tokens) in one forward pass
    delta = compute_delta(token_ids)

    cur, conn = _get_db()
    faiss_index, index_meta = _get_index()

    vec = embed(text)
    faiss_index.add(np.array([vec]))
    index_meta.append(doc_id)

    cur.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?)",
                (doc_id, MODEL_VERSION, msgpack.packb(delta), n, len(delta)))
    conn.commit()

    ratio = 1 - len(delta) / max(n - 1, 1)
    print(f"[WRITE] {doc_id[:8]}... | tokens={n} | Δ={len(delta)} | compression={ratio:.1%}")
    return {"doc_id": doc_id, "token_count": n,
            "surprise_count": len(delta), "compression": ratio}


def mrsf_read(query: str, top_k: int = 1) -> list:
    """Retrieve documents by semantic similarity (O(n) reconstruction).

    Uses ModelSession with KV caching for O(n) token-by-token reconstruction,
    instead of the legacy O(n²) approach.

    Args:
        query : Natural language query
        top_k : Number of results to return

    Returns:
        List of reconstructed text strings
    """
    faiss_index, index_meta = _get_index()
    if faiss_index.ntotal == 0:
        return []

    q_vec = embed(query)
    D, I  = faiss_index.search(np.array([q_vec]), top_k)
    results = []

    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(index_meta):
            continue
        doc_id = index_meta[idx]
        cur, _ = _get_db()
        row = cur.execute(
            "SELECT model_version, delta, token_count FROM documents WHERE doc_id=?",
            (doc_id,)
        ).fetchone()
        if not row:
            continue

        m_ver, delta_blob, token_count = row
        if m_ver != MODEL_VERSION:
            print(f"[WARN] Version mismatch: stored={m_ver} | current={MODEL_VERSION}")

        delta   = {pos: tid for pos, tid in msgpack.unpackb(delta_blob)}
        bos     = tokenize("")[0]
        out_ids = [bos]

        # O(n) reconstruction using ModelSession with KV caching
        session = ModelSession()
        session.feed(bos)

        for i in range(1, token_count):
            if i in delta:
                out_ids.append(delta[i])
            else:
                out_ids.append(session.predict_next())
            session.feed(out_ids[-1])

        # Exclude BOS token when detokenizing
        reconstructed = detokenize(out_ids[1:])

        print(f"[READ]  rank={rank+1} | {doc_id[:8]}... | distance={D[0][rank]:.4f}")
        results.append(reconstructed)

    return results


def save_index():
    """Persist the FAISS index and metadata to disk."""
    faiss_index, index_meta = _get_index()
    faiss.write_index(faiss_index, FAISS_PATH)
    with open(FAISS_PATH + ".meta", "w") as f:
        json.dump(index_meta, f)
    print(f"[INDEX] Saved → {FAISS_PATH}")


def load_index():
    """Load a previously saved FAISS index from disk."""
    global _faiss_index, _index_meta
    if os.path.exists(FAISS_PATH):
        _faiss_index = faiss.read_index(FAISS_PATH)
        with open(FAISS_PATH + ".meta") as f:
            _index_meta = json.load(f)
        print(f"[INDEX] Loaded {_faiss_index.ntotal} documents from disk.")
    else:
        print("[INDEX] No existing index. Starting fresh.")
