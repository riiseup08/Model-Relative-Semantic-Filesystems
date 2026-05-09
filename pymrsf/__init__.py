from .storage import mrsf_write, mrsf_read, save_index, load_index, rebuild_faiss_from_sqlite, close_connections
from .inspect import mrsf_inspect, mrsf_rebuild_explained
from .benchmark import mrsf_benchmark_canterbury, mrsf_latency_benchmark
from .embeddings import embed, get_embedding_dim
from .probe import probe, probe_compare
from .rag import (
    score_chunk, score_chunks, score_chunks_batch,
    explain_chunk, filter_chunks, DEFAULT_WEIGHTS,
    score_chunk_async, score_chunks_async, filter_chunks_async,
)
from .core import (
    ModelSession, compute_delta, get_surprises, tokenize, detokenize,
    quantized_argmax, next_token_greedy,
    get_backend, get_raw_lm, provider_capabilities,
    PROVIDER, MODEL_VERSION, LOGIT_PRECISION,
)
from . import cache
from .cache import (
    configure_cache, get_cache_stats, get_embedding_cache_stats,
    reset_cache_stats, clear_cache, clear_embedding_cache
)

__version__ = "0.4.1"

# ── Temporary backward compatibility aliases (will be removed in future versions) ──
# These allow old code to keep working while migrating to the new API
_get_backend = get_backend  # Deprecated: use get_backend() instead

# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    # Core functions
    "tokenize",
    "detokenize",
    "quantized_argmax",
    "get_surprises",
    "compute_delta",
    "next_token_greedy",
    "ModelSession",
    
    # Backend access
    "get_backend",
    "get_raw_lm",
    "provider_capabilities",
    
    # RAG scoring
    "score_chunk",
    "score_chunks",
    "score_chunks_batch",
    "explain_chunk",
    "filter_chunks",
    "score_chunk_async",
    "score_chunks_async",
    "filter_chunks_async",
    "DEFAULT_WEIGHTS",
    
    # Knowledge probing
    "probe",
    "probe_compare",
    
    # Storage & compression
    "mrsf_write",
    "mrsf_read",
    "save_index",
    "load_index",
    
    # Inspection & debugging
    "mrsf_inspect",
    "mrsf_rebuild_explained",
    
    # Benchmarking
    "mrsf_benchmark_canterbury",
    "mrsf_latency_benchmark",
    
    # Embeddings
    "embed",
    "get_embedding_dim",
    
    # Cache module
    "cache",
    
    # Cache configuration functions
    "configure_cache",
    "get_cache_stats",
    "get_embedding_cache_stats",
    "reset_cache_stats",
    "clear_cache",
    "clear_embedding_cache",
    
    # Constants
    "PROVIDER",
    "MODEL_VERSION",
    "LOGIT_PRECISION",
    
    # Version
    "__version__",
]
