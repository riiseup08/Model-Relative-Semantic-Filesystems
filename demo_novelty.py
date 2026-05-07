#!/usr/bin/env python3
"""
pymrsf demo — see the novelty-aware RAG scoring in action.

This script tests 3 things:
  1. Knowledge probing : Does the model know Shakespeare better than novel text?
  2. RAG scoring       : Do novel+relevant chunks score higher?
  3. Diversity dedup   : Are duplicate chunks filtered out?

Run:  python demo_novelty.py
"""

import sys

# ── Test 1: Knowledge Probe ──────────────────────────────────────────────────

print("=" * 65)
print("TEST 1: Knowledge Probing")
print("Does the model know famous text better than novel text?")
print("=" * 65)

from pymrsf import probe

famous = [
    "To be or not to be, that is the question. Whether tis nobler in the mind to suffer.",
    "The quick brown fox jumps over the lazy dog.",
]

novel = [
    "My proprietary attention mechanism uses a novel sparse gating function.",
    "The Zeta-7 protocol encrypts data using quantum-resistant lattice cryptography.",
]

print("\n📚 FAMOUS TEXT (should score HIGH — model knows this):")
for text in famous:
    result = probe(text)
    if "error" in result:
        print(f"   ⚠️  {result['error']}")
        print(f"   (This is expected if no local model is loaded)")
        break
    bar = "█" * (result['knowledge_score'] // 4) + "░" * (25 - result['knowledge_score'] // 4)
    print(f"   [{bar}] {result['knowledge_score']:>2}/100  {result['label'].upper():<12}  {text[:40]}...")

else:
    print("\n🔬 NOVEL TEXT (should score LOW — model hasn't seen this):")
    for text in novel:
        result = probe(text)
        bar = "█" * (result['knowledge_score'] // 4) + "░" * (25 - result['knowledge_score'] // 4)
        print(f"   [{bar}] {result['knowledge_score']:>2}/100  {result['label'].upper():<12}  {text[:40]}...")

    # Verify the claim
    print(f"\n{'─'*65}")
    famous_avg = sum(probe(t)['knowledge_score'] for t in famous) / len(famous)
    novel_avg  = sum(probe(t)['knowledge_score'] for t in novel) / len(novel)
    print(f"  Famous avg: {famous_avg:.0f}/100  |  Novel avg: {novel_avg:.0f}/100")
    if famous_avg > novel_avg:
        print(f"  ✅ CONFIRMED: Model knows famous text {famous_avg - novel_avg:.0f}% better!")
    else:
        print(f"  ⚠️  Unexpected result — but that's interesting data too!")


# ── Test 2: RAG Chunk Scoring ─────────────────────────────────────────────────

print("\n\n" + "=" * 65)
print("TEST 2: RAG Chunk Scoring")
print("Do novel+relevant chunks score higher than known+irrelevant ones?")
print("=" * 65)

from pymrsf.rag import score_chunks_batch, filter_chunks

# Simulated RAG chunks for the query "How does backpropagation work?"
query = "How does backpropagation work?"
rag_chunks = [
    "Backpropagation computes gradients using the chain rule. It propagates error backwards.",
    "Neural networks are inspired by the human brain. They consist of layers of neurons.",
    "The sky is blue because of Rayleigh scattering. This is well-known physics.",
    "A novel optimization technique uses second-order gradients for faster convergence.",
]

print(f"\nQuery: '{query}'")
print(f"Chunks to score: {len(rag_chunks)}")
print()

results = score_chunks_batch(rag_chunks, query, diversity_threshold=0.90)

for r in results:
    bar = "█" * (r['rag_score'] // 4) + "░" * (25 - r['rag_score'] // 4)
    status = "✅" if r['rag_score'] >= 50 else "❌"
    print(f"  {status} [{bar}] RAG={r['rag_score']:>2}  N={r['novelty_score']:>2}  R={r['relevance_score']:>2}  "
          f"Q={r['query_knowledge']:>2}  {r['verdict'].upper():<10}  {r['chunk'][:50]}...")

best_score = results[0]['rag_score']
worst_score = results[-1]['rag_score']
print(f"\n  Best chunk: {results[0]['chunk'][:50]}... ({best_score}/100)")
print(f"  Worst chunk: {results[-1]['chunk'][:50]}... ({worst_score}/100)")
if best_score > worst_score:
    print(f"  ✅ RAG scoring successfully ranked chunks by usefulness!")
else:
    print(f"  ⚠️  Scores are similar — fine-tuning threshold may help")


# ── Test 3: Diversity Dedup ───────────────────────────────────────────────────

print("\n\n" + "=" * 65)
print("TEST 3: Diversity Dedup")
print("Does the filter remove duplicate chunks?")
print("=" * 65)

# Create chunks where two are very similar
dup_chunks = [
    "Backpropagation computes gradients using the chain rule error propagation.",
    "Backpropagation uses chain rule to compute gradients by propagating error.",
    "The sky is blue because of Rayleigh scattering.",
    "The sky appears blue due to Rayleigh scattering of sunlight.",
]

print(f"\nQuery: 'How does backpropagation work?'")
print(f"Input: {len(dup_chunks)} chunks (2 pairs of near-duplicates)")
print()

# First: score without dedup
print("--- WITHOUT dedup ---")
raw = score_chunks_batch(dup_chunks, query, diversity_threshold=1.0)
for r in raw:
    print(f"  [{r['rag_score']:>2}/100] {r['chunk'][:55]}...")

# Then: with dedup (default threshold 0.85)
print("\n--- WITH dedup (threshold=0.85) ---")
filtered = score_chunks_batch(dup_chunks, query, diversity_threshold=0.85)
dup_count = sum(1 for r in filtered if r['rag_score'] == 0)
for r in filtered:
    if r['rag_score'] > 0:
        print(f"  ✅ [{r['rag_score']:>2}/100] {r['chunk'][:55]}...")
    else:
        print(f"  ❌ [{r['rag_score']:>2}/100] {r['chunk'][:55]}... (DUPLICATE)")

if dup_count > 0:
    print(f"\n  ✅ Dedup removed {dup_count} duplicate chunks!")
else:
    print(f"\n  ℹ️  No duplicates detected (threshold may need adjustment)")


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"\n  pymrsf gives you:")
print(f"  1. Which chunks contain NEW information (not just relevant)")
print(f"  2. Whether the model already KNOWS THE ANSWER to your query")
print(f"  3. Automatic removal of DUPLICATE chunks")
print(f"  4. A tunable RAG score to optimize context window usage")
print(f"\n  Next step: Drop filter_chunks() into your RAG pipeline:")
print(f"    from pymrsf.rag import filter_chunks")
print(f"    good_chunks = filter_chunks(retriever.get(query), query)")
print(f"    answer = llm.complete(query, context=good_chunks)")
print("=" * 65)
