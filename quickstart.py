"""
pymrsf Quick Start — Get started in 5 minutes
==============================================

This file contains minimal, copy-paste-ready examples to get you started.
For comprehensive examples, see examples.py
"""

# ============================================================================
# Setup: Install and Configure
# ============================================================================

"""
# Install pymrsf
pip install pymrsf[local]  # For local models
# OR
pip install pymrsf[openai]  # For OpenAI

# Create .env file
cat > .env << EOF
PYMRSF_PROVIDER=local
PYMRSF_MODEL_PATH=/path/to/mistral-7b-v0.1.Q4_K_M.gguf
PYMRSF_N_GPU_LAYERS=35
EOF
"""

# ============================================================================
# Quick Start 1: Score a Single Chunk
# ============================================================================

from pymrsf import score_chunk

chunk = "Neural networks learn by adjusting weights through backpropagation."
query = "How do neural networks learn?"

result = score_chunk(chunk, query=query)

print(f"RAG Score: {result['rag_score']}/100")
print(f"Verdict: {result['verdict']}")
# Output: RAG Score: 85/100, Verdict: excellent


# ============================================================================
# Quick Start 2: Filter Multiple Chunks
# ============================================================================

from pymrsf import filter_chunks

# Chunks from your vector database
chunks = [
    "Transformers use self-attention to process sequences.",
    "The Eiffel Tower is in Paris, France.",
    "Attention mechanisms allow selective focus on input.",
    "Backpropagation computes gradients for training.",
    "Multi-head attention enables multiple representation subspaces.",
]

query = "How do transformers work?"

# Filter to top 3 relevant chunks
best_chunks = filter_chunks(
    chunks,
    query=query,
    top_k=3,
    min_rag_score=50
)

for i, chunk in enumerate(best_chunks, 1):
    print(f"{i}. {chunk}")

# Output:
# 1. Transformers use self-attention to process sequences.
# 2. Attention mechanisms allow selective focus on input.
# 3. Multi-head attention enables multiple representation subspaces.


# ============================================================================
# Quick Start 3: Complete RAG Pipeline
# ============================================================================

from pymrsf import filter_chunks

def simple_rag_pipeline(query: str, retrieved_chunks: list[str]) -> str:
    """Simple RAG pipeline with pymrsf filtering."""
    
    # Step 1: Filter chunks with pymrsf
    best_chunks = filter_chunks(
        retrieved_chunks,
        query=query,
        top_k=5,
        min_rag_score=50,
        remove_duplicates=True
    )
    
    # Step 2: Build context
    context = "\n\n".join(best_chunks)
    
    # Step 3: Create prompt
    prompt = f"""Answer the question using the context below.

Context:
{context}

Question: {query}

Answer:"""
    
    # Step 4: Send to your LLM (OpenAI, Anthropic, etc.)
    # response = your_llm_client.generate(prompt)
    
    return prompt

# Example usage
query = "What is machine learning?"
retrieved = [
    "Machine learning is AI that learns from data.",
    "Deep learning uses neural networks.",
    "The Eiffel Tower is in Paris.",
]

prompt = simple_rag_pipeline(query, retrieved)
print(prompt)


# ============================================================================
# Quick Start 4: Smart Document Chunking
# ============================================================================

from pymrsf import smart_chunk

document = """
Machine learning is a subset of artificial intelligence. It focuses on 
building systems that learn from data. These systems improve over time 
without explicit programming.

Neural networks are computing systems inspired by biological brains. They 
consist of layers of interconnected nodes. Each connection has a weight 
that adjusts during training.
"""

# Split at semantic boundaries (not arbitrary character limits)
chunks = smart_chunk(
    document,
    min_chunk_len=100,
    max_chunk_len=500,
    target_chunk_len=300
)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\n{chunk}\n")


# ============================================================================
# Quick Start 5: Knowledge Probing (Local Only)
# ============================================================================

from pymrsf import probe, provider_capabilities

# Check if probing is available
if provider_capabilities().get("supports_probe"):
    # Test what the model knows
    result = probe("Paris is the capital of France")
    
    print(f"Known: {result['known']}")  # True
    print(f"Surprise: {result['avg_surprise']:.2f}")  # Low surprise = known
    
    # Test unknown/wrong information
    result = probe("The sky is green")
    print(f"Known: {result['known']}")  # False
    print(f"Surprise: {result['avg_surprise']:.2f}")  # High surprise = novel
else:
    print("Knowledge probing requires local provider")


# ============================================================================
# Quick Start 6: Async for High Throughput
# ============================================================================

import asyncio
from pymrsf.rag import score_chunk_async, filter_chunks_async

async def process_many_chunks():
    """Process multiple chunks concurrently."""
    
    chunks = [
        "Neural networks process information through layers.",
        "Backpropagation trains networks by computing gradients.",
        "Deep learning requires large amounts of data.",
    ]
    
    query = "How do neural networks work?"
    
    # Score all chunks concurrently
    tasks = [score_chunk_async(chunk, query=query) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    for chunk, result in zip(chunks, results):
        print(f"{result['rag_score']}/100: {chunk[:50]}...")

# Run async function
asyncio.run(process_many_chunks())


# ============================================================================
# Quick Start 7: Enable Caching for Speed
# ============================================================================

from pymrsf import cache, score_chunk

# Configure cache
cache.configure_cache(
    enabled=True,
    max_size=10000,
    ttl=3600  # 1 hour
)

# First call: slower (cache miss)
result1 = score_chunk("Some text", query="Some query")

# Second call: much faster (cache hit)
result2 = score_chunk("Some text", query="Some query")

# View cache statistics
cache.print_cache_stats()
# Output: Hits: 1, Misses: 1, Hit Rate: 50.0%


# ============================================================================
# Quick Start 8: Custom Scoring Weights
# ============================================================================

from pymrsf import score_chunk

chunk = "Transformers use attention mechanisms for sequence processing."
query = "What is a transformer?"

# Prioritize novelty (for research/learning)
result = score_chunk(
    chunk,
    query=query,
    novelty_weight=0.8,    # High novelty weight
    relevance_weight=0.2    # Low relevance weight
)
print(f"Novelty-focused: {result['rag_score']}/100")

# Prioritize relevance (for Q&A)
result = score_chunk(
    chunk,
    query=query,
    novelty_weight=0.2,    # Low novelty weight
    relevance_weight=0.8    # High relevance weight
)
print(f"Relevance-focused: {result['rag_score']}/100")


# ============================================================================
# Quick Start 9: CLI Usage
# ============================================================================

"""
# Score a chunk from command line
pymrsf score "Neural networks learn from data" --query "What is machine learning?"

# Probe knowledge
pymrsf probe "The Eiffel Tower is in Paris"

# Check capabilities
pymrsf capabilities
"""


# ============================================================================
# Quick Start 10: Error Handling
# ============================================================================

from pymrsf import score_chunk
import logging

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO)

try:
    result = score_chunk(
        "Your chunk text",
        query="Your query",
        relevance_cutoff=0.7  # Skip if relevance < 0.7
    )
    
    if result['verdict'] == 'skip':
        print("Chunk not relevant enough")
    else:
        print(f"Using chunk: {result['rag_score']}/100")
        
except Exception as e:
    print(f"Error: {e}")
    # Handle error (e.g., model not loaded, API key missing, etc.)


# ============================================================================
# Next Steps
# ============================================================================

"""
📚 Learn More:
- See examples.py for comprehensive examples
- Read README.md for full documentation
- Check CONTRIBUTING.md to contribute

🔧 Configuration:
- Edit .env file for your setup
- See .env.example for all options

💡 Tips:
1. Start with local provider for experimentation
2. Use OpenAI/Anthropic for production scale
3. Enable caching to reduce costs
4. Adjust novelty/relevance weights for your use case
5. Use min_rag_score to control quality threshold

🐛 Troubleshooting:
- Check .env file exists and has correct paths
- Verify model file exists (for local provider)
- Ensure API keys are set (for API providers)
- Check Ollama is running (for embeddings)

🚀 Ready to build amazing RAG systems!
"""
