"""
Example: Using pymrsf with OpenAI API (no local model required)

This example shows how to use pymrsf with the OpenAI provider for
lightweight RAG chunk scoring without downloading a 4GB model.

Setup:
    pip install pymrsf[openai]
    export OPENAI_API_KEY='sk-...'
"""

import os

# Configure to use OpenAI instead of local model
os.environ["PYMRSF_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-...")

from pymrsf.rag import score_chunk, filter_chunks

# Example 1: Score a single chunk
print("=" * 70)
print("Example 1: Scoring a single chunk")
print("=" * 70)

chunk = "Backpropagation computes gradients using the chain rule of calculus."
query = "How does backpropagation work?"

result = score_chunk(chunk, query, verbose=True)

print(f"\nRAG Score: {result['rag_score']}/100")
print(f"Verdict: {result['verdict']}")
print(f"Relevance: {result['relevance_score']}/100")
print(f"Novelty: {result['novelty_score']}/100")

# Example 2: Filter a list of chunks
print("\n" + "=" * 70)
print("Example 2: Filtering multiple chunks")
print("=" * 70)

chunks = [
    "Backpropagation computes gradients using the chain rule.",
    "Neural networks are inspired by the human brain.",
    "The sky is blue because of Rayleigh scattering.",
    "Gradient descent optimizes neural network parameters iteratively.",
    "Paris is the capital of France.",
]

query = "How does backpropagation work?"

print(f"\nQuery: {query}")
print(f"Total chunks: {len(chunks)}")

# Filter to only useful chunks
useful = filter_chunks(
    chunks,
    query,
    min_rag_score=40,  # skip low-value chunks
    top_k=3,            # limit to top 3
    verbose=True,
)

print(f"\nUseful chunks: {len(useful)}")
for i, chunk in enumerate(useful, 1):
    print(f"  {i}. {chunk[:70]}...")

# Example 3: Compare different queries
print("\n" + "=" * 70)
print("Example 3: Query sensitivity")
print("=" * 70)

chunk = "The Transformer architecture uses self-attention mechanisms."

queries = [
    "How do Transformers work?",
    "What is machine learning?",
    "What is the weather in Paris?",
]

for query in queries:
    result = score_chunk(chunk, query)
    print(f"\nQuery: {query}")
    print(f"  RAG Score: {result['rag_score']}/100 ({result['verdict']})")
    print(f"  Relevance: {result['relevance_score']}/100")

# Note about limitations
print("\n" + "=" * 70)
print("Note: OpenAI provider limitations")
print("=" * 70)
print("""
The OpenAI provider uses the logprobs API for basic novelty detection,
but advanced features like knowledge probing require a local model.

For full features:
    pip install pymrsf[local]
    export PYMRSF_PROVIDER=local
    export PYMRSF_MODEL_PATH=./models/mistral-7b-v0.1.Q4_K_M.gguf
""")
