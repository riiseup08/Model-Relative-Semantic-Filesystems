#!/usr/bin/env python3
"""
pymrsf demo – shows write, semantic read, and RAG scoring.
Make sure your .env points to a valid Mistral GGUF model.
"""

import time
from pymrsf import mrsf_write, mrsf_read, probe, save_index
from pymrsf.rag import score_chunk, filter_chunks

# ------------------------------------------------------------
# 1. Write a few documents
# ------------------------------------------------------------
print("=" * 65)
print("PYMRSF DEMO (with KV caching – O(n) reconstruction)")
print("=" * 65)

docs = [
    "The Eiffel Tower is located in Paris, France, and was built in 1889.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Python is a high-level programming language used for data science and machine learning.",
]

print("\n📝 Writing 3 documents...")
for text in docs:
    res = mrsf_write(text)
    print(f"   → {res['doc_id'][:8]}... | {res['token_count']} tokens | "
          f"surprises={res['surprise_count']} | compression={res['compression']:.1%}")

save_index()   # persist FAISS index for later queries

# ------------------------------------------------------------
# 2. Semantic read (retrieve by meaning)
# ------------------------------------------------------------
print("\n" + "─" * 65)
print("🔍 Semantic retrieval")
print("─" * 65)

queries = [
    ("famous iron tower in Paris", "should return Eiffel Tower doc"),
    ("vector similarity search library", "should return FAISS doc"),
    ("high-level language for data science", "should return Python doc"),
]

for query, expected in queries:
    start = time.time()
    results = mrsf_read(query, top_k=1)
    elapsed = (time.time() - start) * 1000   # milliseconds
    if results:
        print(f"\nQuery: '{query}'")
        print(f"  → Retrieved ({elapsed:.1f} ms): {results[0]}")
    else:
        print(f"\nQuery: '{query}' → no results")

# ------------------------------------------------------------
# 3. Probe – how well does the model know a text?
# ------------------------------------------------------------
print("\n" + "─" * 65)
print("🎯 Knowledge probe")
print("─" * 65)

texts_to_probe = [
    "To be or not to be, that is the question.",          # very famous
    "The quick brown fox jumps over the lazy dog.",       # common pangram
    "My proprietary algorithm uses a novel attention mechanism.",  # likely novel
]

for txt in texts_to_probe:
    info = probe(txt)
    print(f"\nText: {txt[:50]}...")
    print(f"  Knowledge score: {info['knowledge_score']}/100  ({info['label']})")
    print(f"  Surprises: {info['surprise_count']} / {info['token_count']-1} tokens")
    if info['surprises']:
        print(f"  Example surprise: '{info['surprises'][0][1]}' at position {info['surprises'][0][0]}")

# ------------------------------------------------------------
# 4. RAG scoring – evaluate chunk usefulness
# ------------------------------------------------------------
print("\n" + "─" * 65)
print("🤖 RAG chunk scoring")
print("─" * 65)

query = "How does backpropagation work?"
chunks = [
    "Backpropagation computes gradients using the chain rule. It propagates error backwards through the network.",
    "Neural networks are inspired by the human brain. They consist of layers of neurons.",
    "The sky is blue because of Rayleigh scattering.",
]

print(f"Query: '{query}'")
for chunk in chunks:
    result = score_chunk(chunk, query)
    print(f"\nChunk: {chunk[:60]}...")
    print(f"  RAG score: {result['rag_score']}/100  ({result['verdict']})")
    print(f"  Novelty: {result['novelty_score']}  |  Relevance: {result['relevance_score']}")

# ------------------------------------------------------------
# 5. Filter chunks for a RAG pipeline
# ------------------------------------------------------------
print("\n" + "─" * 65)
print("🔧 RAG chunk filter (keep only useful chunks)")
print("─" * 65)

good_chunks = filter_chunks(chunks, query, min_rag_score=50, verbose=True)

print(f"\n✅ Kept {len(good_chunks)} chunks for LLM context:")
for i, ch in enumerate(good_chunks, 1):
    print(f"   {i}. {ch}")

print("\n" + "=" * 65)
print("Demo completed. Your pymrsf is working with KV caching!")
print("=" * 65)