# pymrsf Concurrency Model

## Summary

| Component | Thread-safe? | Process-safe? | Notes |
|-----------|-------------|---------------|-------|
| RAG scoring (`score_chunk`, `filter_chunks`, …) | Yes | Yes | Stateless; safe to call from any thread |
| Score cache (`pymrsf.cache`) | Yes | No | Protected by `threading.Lock`; one process only |
| Embedding cache | Yes | No | Same lock model as score cache |
| MRSF storage — reads (`mrsf_read`, `mrsf_read_novel`) | Yes (multiple readers) | No | FAISS search is read-only; safe across threads |
| MRSF storage — writes (`mrsf_write`, `mrsf_delete`) | **No** | No | No internal write lock; external coordination required |
| `rebuild_index` / `save_index` / `load_index` | **No** | No | Mutates global FAISS index; must not overlap with reads |

---

## Supported model: single writer

pymrsf is designed for a **single-writer, multiple-reader** pattern within one process.

- Multiple threads may call `score_chunk`, `filter_chunks`, `embed`, or `mrsf_read` concurrently without coordination.
- Only one thread at a time should call `mrsf_write`, `mrsf_delete`, `rebuild_index`, `save_index`, or `load_index`. Wrap these in a `threading.Lock` if your application has concurrent writers.

```python
import threading, pymrsf

_write_lock = threading.Lock()

def safe_write(text):
    with _write_lock:
        return pymrsf.mrsf_write(text)
```

---

## SQLite WAL mode

The SQLite database (`mrsf.db`) is opened without WAL mode by default, which means concurrent readers block during writes. For read-heavy workloads, enable WAL mode once at startup:

```python
import pymrsf.experimental.storage as store

cur, conn = store._get_db()
cur.execute("PRAGMA journal_mode=WAL")
conn.commit()
```

WAL mode allows concurrent readers while a write is in progress. Writers still serialize against each other.

---

## Multi-process usage

pymrsf does **not** coordinate across processes:

- The in-memory FAISS index and tombstone set are per-process.
- Two processes writing to the same `mrsf.db` / `mrsf.faiss` will corrupt state.

**Recommended pattern for multi-process applications:**

1. Designate one writer process (e.g. an ingestion worker).
2. Reader processes call `load_index()` at startup and periodically thereafter.
3. Use an advisory lock file (e.g. with `fcntl.flock` on Unix or a `.lock` sentinel file on Windows) around all write operations.

```python
import fcntl, pymrsf

def safe_write_multiprocess(text, lock_path="mrsf.lock"):
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        result = pymrsf.mrsf_write(text)
        pymrsf.save_index()
    return result
```

---

## Async support

`score_chunk_async`, `score_chunks_async`, and `filter_chunks_async` use `asyncio.to_thread` to offload CPU-bound scoring. The asyncio event loop itself is single-threaded, so cache reads/writes from async code are safe as long as you do not mix raw `threading` and `asyncio` access to the same cache without coordination.

The incremental async path (`score_chunks_async(incremental=True)`) uses an `asyncio.Lock` to ensure KV-cache state is fed in order — it is sequentially correct but not concurrent across chunks by design.

---

## What is NOT process-safe

- Calling `mrsf_write` from two processes simultaneously.
- Calling `save_index` from one process while another is reading via `mrsf_read`.
- Sharing a single `mrsf.faiss` file between processes without external locking.

If you need true multi-writer distributed storage, use a dedicated vector database (Pinecone, Weaviate, Qdrant, etc.) and use pymrsf for its RAG scoring layer only.
