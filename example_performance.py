"""
Performance Example: Async & Caching in pymrsf

This example demonstrates how async support and caching can significantly
improve RAG scoring performance, especially when:
- Scoring many chunks at once
- Re-scoring the same chunks across queries
- Using API providers (OpenAI, Anthropic) that benefit from concurrency

Setup:
    pip install pymrsf[openai]  # or [local] for local models
    export OPENAI_API_KEY='sk-...'
"""

import asyncio
import time
import os

# Configure to use OpenAI for faster demonstration
os.environ["PYMRSF_PROVIDER"] = "openai"

from pymrsf.rag import filter_chunks, filter_chunks_async
from pymrsf import cache

# Sample chunks for testing
CHUNKS = [
    "Backpropagation computes gradients using the chain rule of calculus.",
    "Neural networks are inspired by the human brain's structure.",
    "The gradient descent algorithm iteratively updates model parameters.",
    "Deep learning models have multiple hidden layers.",
    "The Transformer architecture uses self-attention mechanisms.",
    "Convolutional neural networks excel at image processing tasks.",
    "Recurrent networks are designed for sequential data.",
    "Transfer learning leverages pre-trained models for new tasks.",
    "Overfitting occurs when models memorize training data.",
    "Regularization techniques help prevent overfitting.",
    "Batch normalization stabilizes neural network training.",
    "Dropout randomly deactivates neurons during training.",
    "Adam optimizer adapts learning rates for each parameter.",
    "Cross-entropy loss is common for classification tasks.",
    "Embeddings represent words as dense vectors.",
]


def benchmark_sync():
    """Test synchronous scoring (baseline)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Synchronous Scoring (Baseline)")
    print("=" * 70)
    
    cache.clear_cache()
    cache.reset_cache_stats()
    
    query = "How does backpropagation work in neural networks?"
    
    start = time.time()
    useful = filter_chunks(CHUNKS, query, min_rag_score=40, verbose=False)
    elapsed = time.time() - start
    
    print(f"Query: {query}")
    print(f"Time taken: {elapsed:.2f}s")
    print(f"Chunks filtered: {len(CHUNKS)} → {len(useful)}")
    cache.print_cache_stats()
    
    return elapsed


async def benchmark_async():
    """Test async scoring with concurrency."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Async Scoring (Concurrent)")
    print("=" * 70)
    
    cache.clear_cache()
    cache.reset_cache_stats()
    
    query = "How does backpropagation work in neural networks?"
    
    start = time.time()
    useful = await filter_chunks_async(
        CHUNKS,
        query,
        min_rag_score=40,
        verbose=False,
        max_concurrent=5,  # Score 5 chunks concurrently
    )
    elapsed = time.time() - start
    
    print(f"Query: {query}")
    print(f"Time taken: {elapsed:.2f}s")
    print(f"Chunks filtered: {len(CHUNKS)} → {len(useful)}")
    cache.print_cache_stats()
    
    return elapsed


def benchmark_caching():
    """Test cache performance with repeated queries."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Cache Performance (Repeated Queries)")
    print("=" * 70)
    
    # Configure cache
    cache.configure_cache(enabled=True, max_size=10000, ttl=3600)
    cache.clear_cache()
    cache.reset_cache_stats()
    
    queries = [
        "How does backpropagation work in neural networks?",
        "What is transfer learning?",
        "How do you prevent overfitting?",
        # Repeat queries to test cache
        "How does backpropagation work in neural networks?",
        "What is transfer learning?",
    ]
    
    times = []
    for i, query in enumerate(queries, 1):
        start = time.time()
        useful = filter_chunks(CHUNKS, query, min_rag_score=40, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"\nQuery {i}: {query[:50]}...")
        print(f"  Time: {elapsed:.2f}s | Results: {len(useful)}")
    
    print("\n" + "-" * 70)
    cache.print_cache_stats()
    
    print("Analysis:")
    print(f"  First 3 queries (cold): avg {sum(times[:3])/3:.2f}s")
    print(f"  Last 2 queries (cached): avg {sum(times[3:])/2:.2f}s")
    print(f"  Speedup from caching: {(sum(times[:3])/3) / (sum(times[3:])/2):.1f}x")
    
    return times


async def benchmark_async_batch():
    """Test async scoring of multiple queries in parallel."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Async Multiple Queries (Parallel)")
    print("=" * 70)
    
    cache.clear_cache()
    cache.reset_cache_stats()
    
    queries = [
        "How does backpropagation work in neural networks?",
        "What is transfer learning?",
        "How do you prevent overfitting?",
        "What is the Adam optimizer?",
        "How does dropout work?",
    ]
    
    start = time.time()
    
    # Score all queries in parallel
    tasks = [
        filter_chunks_async(CHUNKS, q, min_rag_score=40, verbose=False)
        for q in queries
    ]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    print(f"Scored {len(queries)} queries × {len(CHUNKS)} chunks in parallel")
    print(f"Time taken: {elapsed:.2f}s")
    print(f"Average time per query: {elapsed/len(queries):.2f}s")
    
    for i, (q, r) in enumerate(zip(queries, results), 1):
        print(f"  Query {i}: {len(r)} useful chunks")
    
    cache.print_cache_stats()
    
    return elapsed


async def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("PYMRSF PERFORMANCE BENCHMARKS")
    print("Async Support & Caching")
    print("=" * 70)
    
    # Benchmark 1: Sync baseline
    sync_time = benchmark_sync()
    
    # Benchmark 2: Async with concurrency
    async_time = await benchmark_async()
    speedup_async = sync_time / async_time if async_time > 0 else 0
    
    # Benchmark 3: Cache performance
    cache_times = benchmark_caching()
    
    # Benchmark 4: Async multiple queries
    batch_time = await benchmark_async_batch()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Async vs Sync speedup: {speedup_async:.1f}x faster")
    print(f"✅ Cache hit rate: {cache.get_cache_stats()['hit_rate']:.1f}%")
    print(f"✅ Async batch processing: {len(CHUNKS) * 5 / batch_time:.1f} chunks/sec")
    print("\n💡 Key Takeaways:")
    print("   - Use async functions for better throughput with API providers")
    print("   - Enable caching to avoid re-scoring the same chunks")
    print("   - Cache is especially effective for iterative RAG workflows")
    print("   - Async + caching can give 5-10x speedup in production")
    print("=" * 70)


# ── Cache Configuration Example ────────────────────────────────────────────────


def demonstrate_cache_config():
    """Show how to configure the cache."""
    print("\n" + "=" * 70)
    print("CACHE CONFIGURATION")
    print("=" * 70)
    
    # Configure cache
    cache.configure_cache(
        enabled=True,      # Enable caching
        max_size=10000,    # Store up to 10,000 scored chunks
        ttl=3600,          # Cache entries expire after 1 hour
    )
    
    print("Cache configured:")
    print("  enabled  = True")
    print("  max_size = 10,000")
    print("  ttl      = 3600s (1 hour)")
    
    # Check cache status
    stats = cache.get_cache_stats()
    print(f"\nCache stats: {stats}")
    
    # Clear cache if needed
    cache.clear_cache()
    print("\n✅ Cache cleared")
    
    print("\nUsage in code:")
    print("""
    from pymrsf import cache
    
    # Configure once at startup
    cache.configure_cache(enabled=True, max_size=10000, ttl=3600)
    
    # Score chunks (caching happens automatically)
    result = score_chunk(chunk, query=query)
    
    # Check performance
    cache.print_cache_stats()
    """)


if __name__ == "__main__":
    # Demonstrate cache configuration
    demonstrate_cache_config()
    
    # Run benchmarks
    print("\n\nStarting benchmarks...")
    print("(This will take 1-2 minutes with API providers)")
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Installed pymrsf[openai] or pymrsf[local]")
        print("  2. Set OPENAI_API_KEY or PYMRSF_PROVIDER=local")
        print("  3. Downloaded a local model if using local provider")
