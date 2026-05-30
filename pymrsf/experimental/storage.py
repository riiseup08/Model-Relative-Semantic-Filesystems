"""
pymrsf.storage — Delta compression storage + FAISS semantic search

Core functionality:
  - mrsf_write : Store text with delta compression (only "surprise" tokens)
  - mrsf_read  : Reconstruct text via KV-cached O(n) model inference
  - mrsf_delete: Remove a document from storage (SQLite + FAISS)
  - save_index : Persist FAISS index to disk
  - load_index : Load persisted FAISS index

FAISS Deletion Strategy:
  FAISS IndexHNSWFlat doesn't support native deletion. Instead we use an
  IDMap wrapper and track deletions via an in-memory tombstone set:
    - On delete: mark doc_id as tombstoned, remove from index_meta
    - On search: filter out tombstoned results
    - Periodic rebuild: reconstruct the index from SQLite to reclaim space

  For production with frequent deletions, run rebuild_index() periodically
  (e.g., every 100 writes or on a cron schedule).
"""

import json
import logging
import os
import sqlite3
import uuid

import faiss
import msgpack
import numpy as np

_logger = logging.getLogger("pymrsf.experimental.storage")

from ..core import (
    MODEL_VERSION,
    ModelSession,
    compute_delta,
    detokenize,
    provider_capabilities,
    tokenize,
)
from ..embeddings import embed, get_embedding_dim

# Configurable paths via environment variables
DB_PATH    = os.getenv("PYMRSF_DB_PATH",    "mrsf.db")
FAISS_PATH = os.getenv("PYMRSF_FAISS_PATH", "mrsf.faiss")
EMBED_DIM  = int(os.getenv("PYMRSF_EMBED_DIM", "768"))  # Validated at runtime

# Lazy initialization — not loaded at import time
_faiss_index = None
_index_meta  = None  # Parallel list: index_meta[i] = doc_id at FAISS position i
_tombstones  = set()  # Set of deleted doc_ids that are still in FAISS index
_conn        = None
_cur         = None
_rebuild_counter = 0  # Counter to trigger periodic rebuilds
_REBUILD_INTERVAL = 100  # Rebuild every N writes


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
                embed_dim      INTEGER,
                deleted        INTEGER DEFAULT 0,
                original_text  TEXT DEFAULT NULL
            )""")
            # Schema migrations for older databases
            for migration in [
                "ALTER TABLE documents ADD COLUMN deleted INTEGER DEFAULT 0",
                "ALTER TABLE documents ADD COLUMN original_text TEXT DEFAULT NULL",
                "ALTER TABLE documents ADD COLUMN marginal_novelty REAL DEFAULT NULL",
            ]:
                try:
                    _cur.execute(migration)
                except sqlite3.OperationalError:
                    pass  # Column already exists
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
                _logger.warning(
                    "Embedding dimension mismatch: expected=%d, actual=%d — using actual.",
                    EMBED_DIM, actual_dim,
                )
                dim = actual_dim
            else:
                dim = EMBED_DIM
            _faiss_index = faiss.IndexHNSWFlat(dim, 32)
            _index_meta  = []
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FAISS index: {e}")
    return _faiss_index, _index_meta


def _add_to_faiss(vec: np.ndarray, doc_id: str):
    """Add a vector to FAISS index with metadata tracking."""
    faiss_index, index_meta = _get_index()
    faiss_index.add(np.array([vec]))
    index_meta.append(doc_id)


def _maybe_rebuild():
    """Periodic automatic index rebuild to reclaim space from tombstones."""
    global _rebuild_counter
    _rebuild_counter += 1
    if _rebuild_counter >= _REBUILD_INTERVAL and len(_tombstones) > 0:
        _rebuild_counter = 0
        rebuild_index(verbose=False)


# ── Public API ─────────────────────────────────────────────────────────────────


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

    vec = embed(text)
    embed_dim = len(vec)

    # If doc_id already exists, delete old entry first
    existing = cur.execute("SELECT doc_id FROM documents WHERE doc_id=? AND deleted=0", (doc_id,)).fetchone()
    if existing:
        mrsf_delete(doc_id)

    # Compute marginal novelty vs. existing corpus before adding to index
    faiss_index, index_meta = _get_index()
    active = sum(1 for m in index_meta if m is not None)
    if active > 0:
        D, _ = faiss_index.search(np.array([vec]), 1)
        cosine_sim = max(0.0, 1.0 - float(D[0][0]) / 2.0)
        marginal_novelty = max(0.0, 1.0 - cosine_sim)
    else:
        marginal_novelty = 1.0  # first document is fully novel

    # Add new vector to FAISS
    _add_to_faiss(vec, doc_id)

    embed_model = os.getenv("PYMRSF_EMBED_MODEL", "nomic-embed-text")

    cur.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (doc_id, MODEL_VERSION, embed_model, msgpack.packb(delta),
                 n, len(delta), len(text), embed_dim, text, marginal_novelty))
    conn.commit()
    _maybe_rebuild()

    ratio = 1 - len(delta) / max(n - 1, 1)
    _logger.debug("WRITE %s... | tokens=%d | Δ=%d | compression=%.1f%%", doc_id[:8], n, len(delta), ratio * 100)
    return {"doc_id": doc_id, "token_count": n,
            "surprise_count": len(delta), "compression": ratio}


def mrsf_delete(doc_id: str) -> bool:
    """Delete a document from storage.

    Marks as deleted in SQLite and adds to FAISS tombstone set.
    The FAISS vector is not immediately removed — space is reclaimed
    on the next rebuild_index() call or automatically after ~100 writes.

    Args:
        doc_id: Document ID to delete

    Returns:
        True if deleted, False if not found
    """
    cur, conn = _get_db()
    cur.execute("UPDATE documents SET deleted=1 WHERE doc_id=? AND deleted=0", (doc_id,))
    conn.commit()

    if cur.rowcount == 0:
        _logger.warning("DELETE %s... not found or already deleted.", doc_id[:8])
        return False

    # Remove from active metadata; keep FAISS vector as tombstone
    _, index_meta = _get_index()
    if doc_id in index_meta:
        idx = index_meta.index(doc_id)
        index_meta[idx] = None  # Mark slot as deleted
        _tombstones.add(doc_id)

    _logger.info("DELETE %s... | tombstoned in FAISS (will be reclaimed on rebuild)", doc_id[:8])
    return True


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
        _logger.error(
            "mrsf_read requires the local provider for ModelSession reconstruction."
            " Install with: pip install pymrsf[local]"
        )
        return []

    faiss_index, index_meta = _get_index()
    # Count non-tombstoned entries
    active_count = sum(1 for m in index_meta if m is not None)
    if active_count == 0:
        return []

    q_vec = embed(query)
    # Search for more results than needed to account for tombstoned entries
    D, indices = faiss_index.search(np.array([q_vec]), min(top_k + len(_tombstones), len(index_meta)))
    results = []

    for rank_idx, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(index_meta):
            continue
        doc_id = index_meta[idx]

        # Skip tombstoned entries (deleted but still in FAISS)
        if doc_id is None or doc_id in _tombstones:
            continue

        cur, _ = _get_db()
        row = cur.execute(
            "SELECT model_version, delta, token_count FROM documents WHERE doc_id=? AND deleted=0",
            (doc_id,)
        ).fetchone()
        if not row:
            continue

        m_ver, delta_blob, token_count = row
        if m_ver != MODEL_VERSION:
            _logger.warning("Version mismatch: stored=%s | current=%s", m_ver, MODEL_VERSION)

        # Safely unpack msgpack
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

        _logger.debug("READ  rank=%d | %s... | distance=%.4f", len(results) + 1, doc_id[:8], D[0][rank_idx])
        results.append(reconstructed)

        if len(results) >= top_k:
            break

    return results


def save_index():
    """Persist the FAISS index and metadata to disk.

    Saves the index as-is, including tombstoned (deleted) vectors in FAISS.
    The metadata file preserves the positional mapping (None for deleted slots).
    To reclaim space from deletions, call rebuild_index() then save_index().

    Note: Full vector recovery on rebuild requires original text, which is not
    stored. For production, store embeddings in SQLite or keep the FAISS file
    as authoritative. Current best practice: avoid frequent deletions, or
    re-add documents after rebuild.
    """
    faiss_index, index_meta = _get_index()

    # Save full index as-is (tombstoned vectors stay in FAISS)
    faiss.write_index(faiss_index, FAISS_PATH)

    # Save metadata preserving positional mapping (None = deleted slot)
    with open(FAISS_PATH + ".meta", "w") as f:
        json.dump(index_meta, f)

    active = sum(1 for m in index_meta if m is not None)
    tombstoned = len(index_meta) - active
    tombstone_note = f" ({tombstoned} tombstoned)" if tombstoned else ""
    _logger.info("INDEX saved → %s (%d active%s)", FAISS_PATH, active, tombstone_note)


def load_index():
    """Load a previously saved FAISS index from disk.

    Reads the FAISS index file and metadata from the paths configured via
    PYMRSF_FAISS_PATH (default: mrsf.faiss) and its .meta companion file.
    If no saved index exists, initializes a fresh empty index.

    Example:
        >>> load_index()
    """
    global _faiss_index, _index_meta, _tombstones
    global _faiss_index, _index_meta, _tombstones
    if os.path.exists(FAISS_PATH):
        try:
            _faiss_index = faiss.read_index(FAISS_PATH)
            with open(FAISS_PATH + ".meta") as f:
                _index_meta = json.load(f)
            # Reset tombstone tracking (all loaded entries are live)
            _tombstones = set()
            _logger.info("INDEX loaded %d documents from disk.", _faiss_index.ntotal)
        except Exception as e:
            _logger.error("Failed to load index: %s — starting with fresh index.", e)
            _faiss_index = None
            _index_meta = None
            _tombstones = set()
            _get_index()  # Initialize fresh
    else:
        _logger.info("INDEX no existing index — starting fresh.")


def reset_index_metadata():
    """Reset FAISS index metadata from SQLite records.

    This rebuilds the index metadata structure (document IDs) from SQLite.
    It does NOT recover original document embeddings — after calling this,
    documents must be re-added via mrsf_write() to populate the FAISS index
    vectors for semantic search.

    Use this after index corruption or when switching to a new embedding
    model that requires a fresh index.
    """
    global _faiss_index, _index_meta, _tombstones

    cur, _ = _get_db()
    rows = cur.execute("SELECT doc_id FROM documents WHERE deleted=0").fetchall()

    if not rows:
        _logger.info("RESET no active documents in SQLite — nothing to reset.")
        return

    _logger.info("RESET rebuilding index metadata from %d SQLite documents...", len(rows))

    # Get fresh index (empty)
    actual_dim = get_embedding_dim()
    _faiss_index = faiss.IndexHNSWFlat(actual_dim, 32)
    _index_meta = []
    _tombstones = set()

    # Rebuild metadata list from SQLite (no vectors recovered)
    for row in rows:
        doc_id = row[0]
        _index_meta.append(doc_id)

    _logger.info(
        "RESET complete — %d index metadata entries."
        " Embedding vectors not recovered; re-add with mrsf_write().",
        len(_index_meta),
    )


def rebuild_index(verbose: bool = True):
    """Rebuild FAISS index from SQLite records, re-embedding from stored original text.

    Documents written with pymrsf >= 0.5.0 have their original text stored in SQLite,
    so their embedding vectors are fully recovered. Legacy rows (original_text=NULL)
    are skipped with a warning — re-add them with mrsf_write() to restore search.

    Args:
        verbose: If True, print progress information

    Returns:
        dict with documents_count, recovered, skipped, and success status
    """
    global _faiss_index, _index_meta, _tombstones, _rebuild_counter

    cur, _ = _get_db()
    rows = cur.execute(
        "SELECT doc_id, original_text FROM documents WHERE deleted=0"
    ).fetchall()

    if not rows:
        if verbose:
            _logger.info("REBUILD no active documents — nothing to rebuild.")
        return {"documents_count": 0, "recovered": 0, "skipped": 0, "success": True}

    if verbose:
        _logger.info("REBUILD rebuilding FAISS index from %d documents...", len(rows))

    actual_dim = get_embedding_dim()
    _faiss_index = faiss.IndexHNSWFlat(actual_dim, 32)
    _index_meta = []
    _tombstones = set()

    recovered = 0
    skipped = 0
    for doc_id, original_text in rows:
        if original_text:
            try:
                vec = embed(original_text)
                _faiss_index.add(np.array([vec]))
                _index_meta.append(doc_id)
                recovered += 1
            except Exception as e:
                if verbose:
                    _logger.warning("REBUILD %s... embed failed: %s — skipping", doc_id[:8], e)
                skipped += 1
        else:
            if verbose:
                _logger.warning("REBUILD %s... no text stored — skipping (re-add with mrsf_write)", doc_id[:8])
            skipped += 1

    _rebuild_counter = 0

    if verbose:
        _logger.info("REBUILD complete — recovered=%d, skipped=%d.", recovered, skipped)
        if skipped:
            _logger.warning("REBUILD %d documents skipped — re-add with mrsf_write() to restore search.", skipped)

    return {"documents_count": recovered, "recovered": recovered, "skipped": skipped, "success": True}


def mrsf_read_novel(
    query: str,
    top_k: int = 5,
    novelty_weight: float = 0.5,
    min_novelty: float = 0.0,
) -> list[dict]:
    """Retrieve documents ranked by a blend of query similarity and corpus novelty.

    Unlike mrsf_read() which ranks purely by embedding similarity, this function
    surfaces documents that are both relevant AND maximally new relative to the
    rest of the corpus (using the marginal_novelty score stored at write time).

    Ranking formula: rank_score = (1 - novelty_weight) * similarity + novelty_weight * marginal_novelty

    Args:
        query          : Natural language query
        top_k          : Number of results to return
        novelty_weight : 0 = pure similarity (same as mrsf_read),
                         1 = pure corpus novelty, 0.5 = balanced (default)
        min_novelty    : Minimum marginal_novelty to include (0=include all)

    Returns:
        List of dicts with keys: text, doc_id, similarity, marginal_novelty, rank_score
    """
    faiss_index, index_meta = _get_index()
    active_count = sum(1 for m in index_meta if m is not None)
    if active_count == 0:
        return []

    q_vec = embed(query)
    search_k = min(top_k * 4 + len(_tombstones), max(active_count, 1))
    D, indices = faiss_index.search(np.array([q_vec]), search_k)

    cur, _ = _get_db()
    candidates = []

    for rank_idx, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(index_meta):
            continue
        doc_id = index_meta[idx]
        if doc_id is None or doc_id in _tombstones:
            continue

        row = cur.execute(
            "SELECT original_text, marginal_novelty FROM documents WHERE doc_id=? AND deleted=0",
            (doc_id,)
        ).fetchone()
        if not row or row[0] is None:
            continue

        original_text, marginal_novelty = row
        # Normalise FAISS L2 distance to approximate cosine similarity
        raw_dist = float(D[0][rank_idx])
        similarity = max(0.0, 1.0 - raw_dist / 2.0)
        novelty = float(marginal_novelty) if marginal_novelty is not None else 0.5

        if novelty < min_novelty:
            continue

        rank_score = (1.0 - novelty_weight) * similarity + novelty_weight * novelty
        candidates.append({
            "text": original_text,
            "doc_id": doc_id,
            "similarity": round(similarity, 4),
            "marginal_novelty": round(novelty, 4),
            "rank_score": round(rank_score, 4),
        })

    candidates.sort(key=lambda x: x["rank_score"], reverse=True)
    return candidates[:top_k]


def close_connections():
    """Close SQLite connection and reset FAISS index state.

    Use this in long-running processes to cleanly release database handles
    and free memory. After calling this, the next mrsf_write or mrsf_read
    call will re-initialize connections lazily.

    Example:
        >>> close_connections()
    """
    global _conn, _cur, _faiss_index, _index_meta, _tombstones
    global _conn, _cur, _faiss_index, _index_meta, _tombstones

    if _conn is not None:
        _conn.close()
        _conn = None
        _cur = None
        _logger.debug("CLEANUP SQLite connection closed.")

    # FAISS index doesn't need explicit cleanup, but we can reset references
    if _faiss_index is not None:
        _logger.debug("CLEANUP FAISS index released (%d documents).", _faiss_index.ntotal)
        _faiss_index = None
        _index_meta = None
        _tombstones = set()
