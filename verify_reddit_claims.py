#!/usr/bin/env python3
"""
Verification script for Reddit post claims about MRSF RAG filtering.
Tests all the specific numbers and claims mentioned in the post.
"""

from pymrsf.probe import probe
from pymrsf.rag import score_chunk, filter_chunks

print("=" * 70)
print("VERIFICATION: Reddit Post Claims")
print("=" * 70)

# Claim 1: Shakespeare quotes score 72/100
print("\n📌 CLAIM 1: Shakespeare quotes score ~72/100 knowledge")
shakespeare = "To be or not to be, that is the question. Whether tis nobler in the mind to suffer."
result = probe(shakespeare)
print(f"   Result: {result['knowledge_score']}/100")
print(f"   Status: {'✅ VERIFIED' if 65 <= result['knowledge_score'] <= 80 else '❌ FAILED'}")

# Claim 2: Novel technical text scores ~29/100
print("\n📌 CLAIM 2: Novel technical text scores ~29/100 knowledge")
novel = "My proprietary attention mechanism uses a novel sparse gating function."
result = probe(novel)
print(f"   Result: {result['knowledge_score']}/100")
print(f"   Status: {'✅ VERIFIED' if 15 <= result['knowledge_score'] <= 40 else '❌ FAILED'}")

# Claim 3: Backpropagation chunk has 50% novelty
print("\n📌 CLAIM 3: Standard backprop chunk has ~50% novelty")
backprop = "Backpropagation computes gradients using the chain rule. This allows neural networks to learn."
query = "How does backpropagation work?"
result = score_chunk(backprop, query)
print(f"   Novelty: {result['novelty_score']}/100")
print(f"   Relevance: {result['relevance_score']}/100")
print(f"   Status: {'✅ VERIFIED' if 40 <= result['novelty_score'] <= 60 else '❌ FAILED'}")

# Claim 4: Irrelevant chunk has low relevance (14%)
print("\n📌 CLAIM 4: Irrelevant sky chunk has low relevance (~14%)")
sky = "The sky is blue because of Rayleigh scattering."
result = score_chunk(sky, query)
print(f"   Relevance: {result['relevance_score']}/100")
print(f"   Status: {'✅ VERIFIED' if result['relevance_score'] <= 20 else '❌ FAILED'}")

# Claim 5: Novel optimization chunk scores 51/100 RAG score
print("\n📌 CLAIM 5: Novel optimization chunk scores ~51/100 RAG")
novel_opt = "A novel optimization technique uses second-order gradients for faster convergence."
result = score_chunk(novel_opt, query)
print(f"   RAG Score: {result['rag_score']}/100")
print(f"   Novelty: {result['novelty_score']}/100")
print(f"   Relevance: {result['relevance_score']}/100")
print(f"   Status: {'✅ VERIFIED' if 45 <= result['rag_score'] <= 60 else '❌ FAILED'}")

# Claim 6: filter_chunks works as drop-in
print("\n📌 CLAIM 6: filter_chunks works as drop-in replacement")
chunks = [
    "Backpropagation computes gradients using the chain rule.",
    "Neural networks are inspired by the human brain.",
    "The sky is blue because of Rayleigh scattering.",
    "A novel optimization technique uses second-order gradients.",
]
try:
    good_chunks = filter_chunks(chunks, query, min_rag_score=50, top_k=5)
    print(f"   Input chunks: {len(chunks)}")
    print(f"   Filtered chunks: {len(good_chunks)}")
    print(f"   Status: ✅ VERIFIED (function works)")
except Exception as e:
    print(f"   Status: ❌ FAILED ({e})")

# Claim 7: Duplicate detection works
print("\n📌 CLAIM 7: Duplicate detection removes similar chunks")
chunks_with_dupes = [
    "Backpropagation uses chain rule to compute gradients by propagating errors backward.",
    "The sky appears blue due to Rayleigh scattering of sunlight.",
    "Backpropagation computes gradients using the chain rule.",  # Near duplicate
    "The sky is blue because of Rayleigh scattering.",  # Near duplicate
]
try:
    # With diversity_threshold=1.0, no dedup happens
    no_dedup = filter_chunks(chunks_with_dupes, query, min_rag_score=1, diversity_threshold=1.0)
    # With diversity_threshold=0.85, duplicates get rag_score=0 and are filtered by min_rag_score=1
    with_dedup = filter_chunks(chunks_with_dupes, query, min_rag_score=1, diversity_threshold=0.85)
    removed = len(no_dedup) - len(with_dedup)
    print(f"   Without dedup: {len(no_dedup)} chunks")
    print(f"   With dedup: {len(with_dedup)} chunks")
    print(f"   Removed: {removed} duplicates")
    print(f"   Status: {'✅ VERIFIED' if removed > 0 else '❌ FAILED'}")
except Exception as e:
    print(f"   Status: ❌ FAILED ({e})")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
