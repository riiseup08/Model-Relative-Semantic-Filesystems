"""
pymrsf — Comprehensive Examples
================================

This file demonstrates all major features of pymrsf with practical,
production-ready examples.
"""

# ============================================================================
# Example 1: Basic RAG Chunk Scoring
# ============================================================================

def example_basic_scoring():
    """Score individual chunks for relevance and novelty."""
    from pymrsf import score_chunk
    
    # Sample chunks retrieved from a vector database
    chunks = [
        "Neural networks are computing systems inspired by biological neural networks. "
        "They consist of interconnected nodes called neurons.",
        
        "Backpropagation is the primary algorithm for training neural networks. "
        "It computes gradients of the loss function with respect to weights.",
        
        "The Eiffel Tower is a famous landmark in Paris, France.",
    ]
    
    query = "How do neural networks learn?"
    
    print("=" * 60)
    print("Example 1: Basic Chunk Scoring")
    print("=" * 60)
    print(f"Query: {query}\n")
    
    for i, chunk in enumerate(chunks, 1):
        result = score_chunk(chunk, query=query)
        
        print(f"Chunk {i}:")
        print(f"  Text: {chunk[:60]}...")
        print(f"  RAG Score: {result['rag_score']}/100")
        print(f"  Verdict: {result['verdict']}")
        print(f"  Novelty: {result.get('novelty_score', 'N/A')}")
        print(f"  Relevance: {result.get('relevance_score', 'N/A')}")
        print()


# ============================================================================
# Example 2: Filtering and Ranking Chunks
# ============================================================================

def example_chunk_filtering():
    """Filter and rank chunks for optimal RAG context."""
    from pymrsf import filter_chunks
    
    # Simulated retrieval results (would come from your vector DB)
    retrieved_chunks = [
        "Transformers use self-attention mechanisms to process sequences.",
        "The attention mechanism allows models to focus on relevant parts of input.",
        "BERT is a transformer-based model for natural language understanding.",
        "Transformers have revolutionized NLP since their introduction in 2017.",
        "The self-attention mechanism computes weighted representations of input tokens.",
        "GPT models are autoregressive transformers trained on next-token prediction.",
        "Transformers can process entire sequences in parallel, unlike RNNs.",
        "The Eiffel Tower was built for the 1889 World's Fair.",
        "Attention is computed using queries, keys, and values.",
        "Multi-head attention allows the model to attend to different aspects.",
    ]
    
    query = "How does the attention mechanism work in transformers?"
    
    print("=" * 60)
    print("Example 2: Filtering and Ranking Chunks")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Retrieved chunks: {len(retrieved_chunks)}\n")
    
    # Filter to top 5 most relevant and novel chunks
    best_chunks = filter_chunks(
        retrieved_chunks,
        query=query,
        min_rag_score=50,  # Only keep chunks scoring above 50
        top_k=5,           # Keep top 5 chunks
        remove_duplicates=True  # Remove semantically similar chunks
    )
    
    print(f"Filtered to {len(best_chunks)} chunks:\n")
    for i, chunk in enumerate(best_chunks, 1):
        print(f"{i}. {chunk[:80]}...")
    
    return best_chunks


# ============================================================================
# Example 3: Smart Chunking with Semantic Boundaries
# ============================================================================

def example_smart_chunking():
    """Split documents at semantic boundaries using model surprise."""
    from pymrsf import smart_chunk
    
    # Long document to chunk
    document = """
    Machine learning is a subset of artificial intelligence that focuses on 
    developing systems that can learn from data. The key advantage of machine 
    learning is that systems can improve their performance without being 
    explicitly programmed for every scenario.
    
    Neural networks are a fundamental architecture in deep learning. They consist
    of layers of interconnected nodes that process information. Each connection 
    has an associated weight that gets adjusted during training.
    
    The training process uses backpropagation to compute gradients. These gradients
    indicate how much each weight should be adjusted to minimize the loss function.
    The optimization algorithm, such as Adam or SGD, then updates the weights.
    
    Transfer learning has become increasingly important in modern AI. Instead of
    training from scratch, models can be pre-trained on large datasets and then
    fine-tuned for specific tasks. This approach saves computational resources
    and often achieves better performance.
    """
    
    print("=" * 60)
    print("Example 3: Smart Chunking")
    print("=" * 60)
    print(f"Document length: {len(document)} characters\n")
    
    # Smart chunking finds natural semantic boundaries
    chunks = smart_chunk(
        document,
        min_chunk_len=100,   # Minimum chunk size
        max_chunk_len=500,   # Maximum chunk size
        target_chunk_len=300 # Target chunk size
    )
    
    print(f"Created {len(chunks)} semantic chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(f"  {chunk[:100].strip()}...")
        print()


# ============================================================================
# Example 4: Knowledge Probing
# ============================================================================

def example_knowledge_probing():
    """Probe what the model already knows."""
    from pymrsf import probe, provider_capabilities
    
    print("=" * 60)
    print("Example 4: Knowledge Probing")
    print("=" * 60)
    
    # Check if probing is supported
    if not provider_capabilities().get("supports_probe", False):
        print("⚠️  Knowledge probing requires local provider")
        print("   Set PYMRSF_PROVIDER=local in .env")
        return
    
    test_statements = [
        "The capital of France is Paris.",
        "Quantum entanglement allows instantaneous communication.",
        "The Riemann hypothesis was proven in 2023.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]
    
    for statement in test_statements:
        result = probe(statement)
        
        print(f"\nStatement: {statement}")
        print(f"  Known: {result.get('known', 'N/A')}")
        print(f"  Avg Surprise: {result.get('avg_surprise', 'N/A'):.2f}")
        print(f"  Confidence: {'High' if result.get('known') else 'Low'}")


# ============================================================================
# Example 5: Complete RAG Pipeline
# ============================================================================

def example_complete_rag_pipeline():
    """Complete RAG pipeline with all features."""
    from pymrsf import filter_chunks, score_chunk
    import numpy as np
    
    print("=" * 60)
    print("Example 5: Complete RAG Pipeline")
    print("=" * 60)
    
    # Step 1: Simulate document retrieval
    query = "What are the key components of a transformer architecture?"
    
    print(f"Query: {query}\n")
    print("Step 1: Retrieving documents from vector database...")
    
    # In production, this would come from your vector DB
    retrieved_docs = [
        "The transformer architecture consists of an encoder and decoder.",
        "Self-attention is the core mechanism in transformers.",
        "Transformers use positional encodings to capture sequence order.",
        "Multi-head attention allows focusing on different representation subspaces.",
        "The feed-forward network in each layer processes attended representations.",
        "Layer normalization stabilizes training in transformer models.",
        "The encoder processes the input sequence into continuous representations.",
        "The decoder generates output sequences autoregressively.",
        "Residual connections help gradients flow through deep transformer layers.",
        "The key-query-value mechanism enables selective attention.",
    ]
    
    print(f"  Retrieved {len(retrieved_docs)} documents\n")
    
    # Step 2: Score and filter chunks
    print("Step 2: Scoring and filtering chunks...")
    filtered = filter_chunks(
        retrieved_docs,
        query=query,
        min_rag_score=40,
        top_k=5,
        remove_duplicates=True
    )
    print(f"  Kept {len(filtered)} high-quality chunks\n")
    
    # Step 3: Show final context
    print("Step 3: Final context for LLM:")
    print("-" * 60)
    context = "\n\n".join(filtered)
    print(context)
    print("-" * 60)
    
    # Step 4: Construct prompt (would be sent to your LLM)
    prompt = f"""Answer the following question using the provided context.

Context:
{context}

Question: {query}

Answer:"""
    
    print("\nStep 4: Prompt ready for LLM generation")
    print(f"  Context length: {len(context)} chars")
    print(f"  Prompt length: {len(prompt)} chars")


# ============================================================================
# Example 6: Async Operations for High Throughput
# ============================================================================

async def example_async_operations():
    """Process multiple chunks concurrently with async API."""
    import asyncio
    from pymrsf.rag import score_chunk_async, filter_chunks_async
    
    print("=" * 60)
    print("Example 6: Async Operations")
    print("=" * 60)
    
    chunks = [
        "Deep learning models require large amounts of training data.",
        "Convolutional neural networks excel at image processing tasks.",
        "Recurrent neural networks can process sequential data.",
        "Attention mechanisms allow models to focus selectively.",
        "Transfer learning reduces training time significantly.",
    ]
    
    query = "What are the advantages of deep learning?"
    
    print(f"Query: {query}")
    print(f"Processing {len(chunks)} chunks asynchronously...\n")
    
    # Score all chunks concurrently
    tasks = [score_chunk_async(chunk, query=query) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    for i, (chunk, result) in enumerate(zip(chunks, results), 1):
        print(f"Chunk {i}: {result['rag_score']}/100 - {result['verdict']}")
    
    print("\nAsync filtering...")
    filtered = await filter_chunks_async(chunks, query=query, top_k=3)
    print(f"Filtered to {len(filtered)} chunks")


# ============================================================================
# Example 7: Caching for Performance
# ============================================================================

def example_caching():
    """Use caching to improve performance for repeated queries."""
    from pymrsf import cache, score_chunk
    import time
    
    print("=" * 60)
    print("Example 7: Caching")
    print("=" * 60)
    
    # Configure cache
    cache.configure_cache(
        enabled=True,
        max_size=1000,
        ttl=3600  # 1 hour
    )
    
    chunk = "Neural networks learn by adjusting weights through backpropagation."
    query = "How do neural networks learn?"
    
    # First call - cache miss
    print("First call (cache miss)...")
    start = time.time()
    result1 = score_chunk(chunk, query=query)
    time1 = time.time() - start
    print(f"  Time: {time1:.3f}s")
    print(f"  Score: {result1['rag_score']}")
    
    # Second call - cache hit
    print("\nSecond call (cache hit)...")
    start = time.time()
    result2 = score_chunk(chunk, query=query)
    time2 = time.time() - start
    print(f"  Time: {time2:.3f}s")
    print(f"  Score: {result2['rag_score']}")
    print(f"  Speedup: {time1/time2:.1f}x")
    
    # Show cache stats
    print("\nCache Statistics:")
    cache.print_cache_stats()


# ============================================================================
# Example 8: Custom Scoring Weights
# ============================================================================

def example_custom_weights():
    """Customize novelty and relevance weights for specific use cases."""
    from pymrsf import score_chunk
    
    print("=" * 60)
    print("Example 8: Custom Scoring Weights")
    print("=" * 60)
    
    chunk = "Transformers use self-attention to process sequences in parallel."
    query = "What is a transformer?"
    
    # Scenario 1: Prioritize novelty (e.g., for research/learning)
    print("Scenario 1: High novelty weight (research/learning)")
    result = score_chunk(
        chunk,
        query=query,
        novelty_weight=0.8,
        relevance_weight=0.2
    )
    print(f"  RAG Score: {result['rag_score']}/100")
    print(f"  Verdict: {result['verdict']}\n")
    
    # Scenario 2: Prioritize relevance (e.g., for question answering)
    print("Scenario 2: High relevance weight (Q&A)")
    result = score_chunk(
        chunk,
        query=query,
        novelty_weight=0.2,
        relevance_weight=0.8
    )
    print(f"  RAG Score: {result['rag_score']}/100")
    print(f"  Verdict: {result['verdict']}\n")
    
    # Scenario 3: Balanced (default)
    print("Scenario 3: Balanced weights (general purpose)")
    result = score_chunk(
        chunk,
        query=query,
        novelty_weight=0.6,
        relevance_weight=0.4
    )
    print(f"  RAG Score: {result['rag_score']}/100")
    print(f"  Verdict: {result['verdict']}")


# ============================================================================
# Example 9: Multi-Provider Setup
# ============================================================================

def example_multi_provider():
    """Compare behavior across different providers."""
    from pymrsf import score_chunk, provider_capabilities
    import os
    
    print("=" * 60)
    print("Example 9: Multi-Provider Setup")
    print("=" * 60)
    
    current_provider = os.getenv("PYMRSF_PROVIDER", "local")
    print(f"Current provider: {current_provider}\n")
    
    # Show capabilities
    caps = provider_capabilities()
    print("Provider capabilities:")
    for feature, supported in caps.items():
        status = "✓" if supported else "✗"
        print(f"  {status} {feature}")
    
    print("\nProvider-specific recommendations:")
    if current_provider == "local":
        print("  • Full feature support")
        print("  • Use probe() for knowledge detection")
        print("  • Use smart_chunk() for semantic chunking")
        print("  • Best for: research, experimentation, privacy")
    elif current_provider == "openai":
        print("  • Limited surprise signal (via logprobs)")
        print("  • Best for: production, scalability")
        print("  • Consider: caching for cost optimization")
    elif current_provider == "anthropic":
        print("  • No surprise signal (embedding-based only)")
        print("  • Best for: Claude-specific applications")
        print("  • Falls back to embedding similarity")


# ============================================================================
# Example 10: Production-Ready RAG System
# ============================================================================

class ProductionRAGSystem:
    """
    Production-ready RAG system using pymrsf.
    
    Features:
    - Configurable scoring parameters
    - Automatic caching
    - Error handling
    - Logging
    - Metrics collection
    """
    
    def __init__(
        self,
        min_rag_score: float = 50.0,
        top_k: int = 5,
        novelty_weight: float = 0.6,
        relevance_weight: float = 0.4,
        enable_cache: bool = True
    ):
        from pymrsf import cache
        import logging
        
        self.min_rag_score = min_rag_score
        self.top_k = top_k
        self.novelty_weight = novelty_weight
        self.relevance_weight = relevance_weight
        
        # Setup logging
        self.logger = logging.getLogger("ProductionRAG")
        self.logger.setLevel(logging.INFO)
        
        # Setup caching
        if enable_cache:
            cache.configure_cache(enabled=True, max_size=10000, ttl=3600)
            self.logger.info("Cache enabled")
        
        # Metrics
        self.metrics = {
            "queries_processed": 0,
            "chunks_scored": 0,
            "chunks_filtered": 0,
        }
    
    def process_query(self, query: str, retrieved_chunks: list[str]) -> dict:
        """
        Process a RAG query with retrieved chunks.
        
        Args:
            query: User query
            retrieved_chunks: Chunks from vector DB
        
        Returns:
            dict with filtered_chunks, metrics, and metadata
        """
        from pymrsf import filter_chunks
        import time
        
        start_time = time.time()
        self.logger.info(f"Processing query: {query[:50]}...")
        
        try:
            # Filter chunks
            filtered = filter_chunks(
                retrieved_chunks,
                query=query,
                min_rag_score=self.min_rag_score,
                top_k=self.top_k,
                remove_duplicates=True,
                novelty_weight=self.novelty_weight,
                relevance_weight=self.relevance_weight
            )
            
            # Update metrics
            self.metrics["queries_processed"] += 1
            self.metrics["chunks_scored"] += len(retrieved_chunks)
            self.metrics["chunks_filtered"] += len(filtered)
            
            processing_time = time.time() - start_time
            
            result = {
                "filtered_chunks": filtered,
                "num_input_chunks": len(retrieved_chunks),
                "num_output_chunks": len(filtered),
                "processing_time": processing_time,
                "query": query
            }
            
            self.logger.info(
                f"Filtered {len(retrieved_chunks)} -> {len(filtered)} chunks "
                f"in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise
    
    def get_metrics(self) -> dict:
        """Get system metrics."""
        from pymrsf import cache
        
        metrics = self.metrics.copy()
        
        # Add cache metrics
        cache_stats = cache.get_cache_stats()
        metrics["cache_stats"] = cache_stats
        
        # Calculate derived metrics
        if metrics["queries_processed"] > 0:
            metrics["avg_chunks_per_query"] = (
                metrics["chunks_scored"] / metrics["queries_processed"]
            )
            metrics["avg_filtered_per_query"] = (
                metrics["chunks_filtered"] / metrics["queries_processed"]
            )
            metrics["filter_ratio"] = (
                metrics["chunks_filtered"] / metrics["chunks_scored"]
            )
        
        return metrics


def example_production_system():
    """Demonstrate production-ready RAG system."""
    print("=" * 60)
    print("Example 10: Production RAG System")
    print("=" * 60)
    
    # Initialize system
    rag_system = ProductionRAGSystem(
        min_rag_score=50.0,
        top_k=5,
        enable_cache=True
    )
    
    # Simulate multiple queries
    queries = [
        ("What is machine learning?", [
            "Machine learning is a subset of AI focused on learning from data.",
            "Deep learning uses neural networks with multiple layers.",
            "Supervised learning requires labeled training data.",
        ]),
        ("How do transformers work?", [
            "Transformers use self-attention mechanisms.",
            "The attention mechanism computes weighted representations.",
            "Multi-head attention processes different representation subspaces.",
        ]),
    ]
    
    print("Processing queries...\n")
    for query, chunks in queries:
        result = rag_system.process_query(query, chunks)
        print(f"Query: {query}")
        print(f"  Input: {result['num_input_chunks']} chunks")
        print(f"  Output: {result['num_output_chunks']} chunks")
        print(f"  Time: {result['processing_time']:.3f}s\n")
    
    # Show metrics
    print("System Metrics:")
    print("-" * 60)
    metrics = rag_system.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


# ============================================================================
# Main: Run all examples
# ============================================================================

def main():
    """Run all examples."""
    import sys
    import asyncio
    
    examples = [
        ("Basic Scoring", example_basic_scoring),
        ("Chunk Filtering", example_chunk_filtering),
        ("Smart Chunking", example_smart_chunking),
        ("Knowledge Probing", example_knowledge_probing),
        ("Complete RAG Pipeline", example_complete_rag_pipeline),
        ("Async Operations", example_async_operations),
        ("Caching", example_caching),
        ("Custom Weights", example_custom_weights),
        ("Multi-Provider", example_multi_provider),
        ("Production System", example_production_system),
    ]
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "pymrsf Examples" + " " * 28 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # If argument provided, run specific example
    if len(sys.argv) > 1:
        try:
            idx = int(sys.argv[1]) - 1
            if 0 <= idx < len(examples):
                name, func = examples[idx]
                print(f"Running Example {idx + 1}: {name}\n")
                if asyncio.iscoroutinefunction(func):
                    asyncio.run(func())
                else:
                    func()
            else:
                print(f"Invalid example number. Choose 1-{len(examples)}")
        except ValueError:
            print("Usage: python examples.py [example_number]")
    else:
        # Run all examples
        print("Running all examples...\n")
        for i, (name, func) in enumerate(examples, 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    asyncio.run(func())
                else:
                    func()
                print("\n")
            except Exception as e:
                print(f"❌ Example {i} failed: {e}\n")
    
    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
