import sqlite3, msgpack, uuid, json, os
import numpy as np
import faiss

from .core import lm, tokenize, detokenize, quantized_argmax, next_token_greedy, MODEL_VERSION
from .embeddings import embed

DB_PATH    = "mrsf.db"
FAISS_PATH = "mrsf.faiss"
EMBED_DIM  = 768

faiss_index = faiss.IndexHNSWFlat(EMBED_DIM, 32)
index_meta  = []

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    model_version  TEXT,
    delta          BLOB,
    token_count    INTEGER,
    surprise_count INTEGER
)""")
conn.commit()


def mrsf_write(text: str, doc_id: str = None) -> dict:
    doc_id    = doc_id or str(uuid.uuid4())
    token_ids = tokenize(text)          # includes BOS
    n         = len(token_ids)

    lm.reset()
    lm.eval(token_ids)

    delta = []
    for i in range(n - 1):
        pred   = quantized_argmax(np.array(lm.scores[i]))
        actual = token_ids[i + 1]
        if pred != actual:
            delta.append((i + 1, actual))

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
    if faiss_index.ntotal == 0:
        return ["No documents stored yet."]

    q_vec = embed(query)
    D, I  = faiss_index.search(np.array([q_vec]), top_k)
    results = []

    for rank, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(index_meta):
            continue
        doc_id = index_meta[idx]
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
        
        for i in range(1, token_count):
            if i in delta:
                out_ids.append(delta[i])
            else:
                out_ids.append(next_token_greedy(out_ids))
        
        reconstructed = detokenize(out_ids)
        
        print(f"[READ]  rank={rank+1} | {doc_id[:8]}... | distance={D[0][rank]:.4f}")
        results.append(reconstructed)

    return results


def save_index():
    faiss.write_index(faiss_index, FAISS_PATH)
    with open(FAISS_PATH + ".meta", "w") as f:
        json.dump(index_meta, f)
    print(f"[INDEX] Saved → {FAISS_PATH}")


def load_index():
    global faiss_index, index_meta
    if os.path.exists(FAISS_PATH):
        faiss_index = faiss.read_index(FAISS_PATH)
        with open(FAISS_PATH + ".meta") as f:
            index_meta.extend(json.load(f))
        print(f"[INDEX] Loaded {faiss_index.ntotal} documents from disk.")
    else:
        print("[INDEX] No existing index. Starting fresh.")