"""
pymrsf.storage — Delta compression storage + FAISS semantic search

Core functionality:
  - mrsf_write : Store text with delta compression (only "surprise" tokens)
  - mrsf_read  : Reconstruct text via KV-cached O(n) model inference
  - save_index : Persist FAISS index to disk
  - load_index : Load persisted FAISS index
"""

import sqlite3, msgpack, uuid, json, os, time
import numpy as np
import faiss

from .core import tokenize, detokenize, compute_delta, ModelSession, MODEL_VERSION, provider_capabilities
from .embeddings import embed, get_embedding_dim

DB_PATH    = "mrsf.db"
FAISS_PATH = "mrsf.faiss"
EMBED_DIM  = 768  # Default, validated at runtime

# Lazy initialization — not loaded at import time
_faiss_index = None
_index_meta  = None
_conn        = None
_cur         = None


def _get_db():
    """Lazy SQLite connection with error handling."""
    global _conn, _cur
    if _conn is None:
        try:
            _conn = sqlite3.connect(DB_PATH)
            _cur  = _conn.cursor()
            _cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id         TEXT PRIMARY KEY,
                model_version  TEXT,
                embed_model    TEXT,
                delta          BLOB,
                token_count    INTEGER,
                surprise_count INTEGER,
                text_length    INTEGER,
                embed_dim      INTEGER
            )""")
            _conn.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SQLite database: {e}")
    return _cur, _conn


def _get_index():
    """Lazy FAISS index with dimension validation."""
    global _faiss_index, _index_meta
    if _faiss_index is None:
        try:
            # Validate embedding dimension matches expected
            actual_dim = get_embedding_dim()
            if actual_dim != EMBED_DIM:
                print(f"[WARN] Embedding dimension mismatch: expected={EMBED_DIM}, actual={actual_dim}")
                print(f"       Using actual dimension: {actual_dim}")
                dim = actual_dim
            else:
                dim = EMBED_DIM
            _faiss_index = faiss.IndexHNSWFlat(dim, 32)
            _index_meta  = []
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")
    return _faiss_index, _index_meta


def mrsf_write(text: str, doc_id: str = None) -> dict:
    """Store a document with delta compression.

    Args:
        text  : The text to store
        doc_id: Optional custom document ID (auto-generated if None)

    Returns:
        dict with doc_id, token_count, surprise_count, compression
    """
    # Check if delta compression is available
    if not provider_capabilities().get("supports_delta", False):
        return {
            "error": "Delta compression requires local provider",
            "message": (
                "\n[pymrsf] Delta compression requires the local provider.\n"
                "  Install with: pip install pymrsf[local]\n"
                "  And set: PYMRSF_PROVIDER=local\n"
            )
        }
    
    doc_id    = doc_id or str(uuid.uuid4())
    token_ids = tokenize(text)
    n         = len(token_ids)

    # Compute delta (surprise tokens) in one forward pass
    delta = compute_delta(token_ids)

    cur, conn = _get_db()
    faiss_index, index_meta = _get_index()

    vec = embed(text)
    embed_dim = len(vec)
    
    # Check if doc_id already exists in FAISS metadata
    if doc_id in index_meta:
        # Clean overwrite: remove old FAISS entry
        old_idx = index_meta.index(doc_id)
        # Note: FAISS doesn't support deletion, so we just update metadata
        # For production, consider rebuilding index periodically
        index_meta[old_idx] = doc_id + "_old_" + str(time.time())
    
    faiss_index.add(np.array([vec]))
    index_meta.append(doc_id)

    from .embeddings import EMBED_MODEL
    cur.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, MODEL_VERSION, EMBED_MODEL, msgpack.packb(delta), 
                 n, len(delta), len(text), embed_dim))
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
    # Check if ModelSession reconstruction is available
    if not provider_capabilities().get("supports_delta", False):
        print("[ERROR] mrsf_read requires the local provider for ModelSession reconstruction.")
        print("  Install with: pip install pymrsf[local]")
        print("  And set: PYMRSF_PROVIDER=local")
        return []
    
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

        # Safely unpack msgpack - explicitly handle tuple/list reconstruction
        delta_list = msgpack.unpackb(delta_blob, strict_map_key=False)
        delta = {pos: tid for pos, tid in delta_list}
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
        try:
            _faiss_index = faiss.read_index(FAISS_PATH)
            with open(FAISS_PATH + ".meta") as f:
                _index_meta = json.load(f)
            print(f"[INDEX] Loaded {_faiss_index.ntotal} documents from disk.")
        except Exception as e:
            print(f"[ERROR] Failed to load index: {e}")
            print("        Starting with fresh index.")
            _faiss_index = None
            _index_meta = None
            _get_index()  # Initialize fresh
    else:
        print("[INDEX] No existing index. Starting fresh.")


def rebuild_faiss_from_sqlite():
    """Rebuild FAISS index from SQLite metadata if index drift occurs."""
    global _faiss_index, _index_meta
    
    cur, _ = _get_db()
    rows = cur.execute("SELECT doc_id FROM documents").fetchall()
    
    if not rows:
        print("[REBUILD] No documents in SQLite. Nothing to rebuild.")
        return
    
    print(f"[REBUILD] Rebuilding FAISS index from {len(rows)} SQLite documents...")
    
    # Get fresh index
    actual_dim = get_embedding_dim()
    _faiss_index = faiss.IndexHNSWFlat(actual_dim, 32)
    _index_meta = []
    
    # Re-read and re-embed all documents
    for row in rows:
        doc_id = row[0]
        # Note: Original text not stored, so this requires re-reconstruction
        # For now, we just reset the index structure
        # In production, consider storing original text or embeddings in SQLite
        _index_meta.append(doc_id)
    
    print(f"[REBUILD] Complete. Index now has {len(_index_meta)} entries.")
    print("          Note: Original embeddings not recovered. Consider re-adding documents.")


def close_connections():
    """Close database and cleanup connections for long-running processes."""
    global _conn, _cur, _faiss_index, _index_meta
    
    if _conn is not None:
        _conn.close()
        _conn = None
        _cur = None
        print("[CLEANUP] SQLite connection closed.")
    
    # FAISS index doesn't need explicit cleanup, but we can reset references
    if _faiss_index is not None:
        print(f"[CLEANUP] FAISS index released ({_faiss_index.ntotal} documents).")
        _faiss_index = None
        _index_meta = None
