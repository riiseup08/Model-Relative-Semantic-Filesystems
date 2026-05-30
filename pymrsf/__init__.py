"""
pymrsf — Model-Relative Semantic Filtering

Use language model surprise signals for smarter RAG pipelines:

    from pymrsf import score_chunk, filter_chunks, smart_chunk, probe

    # Score a candidate chunk against a query
    result = score_chunk(chunk_text, query="What is backpropagation?")

    # Filter a list of chunks to the most relevant
    kept = filter_chunks(chunks, query="neural network training")

    # Split text at semantic boundaries (surprise-guided)
    pieces = smart_chunk(long_document)

    # Probe what the model already knows
    knowledge = probe("What is the Eiffel Tower?")

The MRSF delta-compression storage system (mrsf_write, mrsf_read, etc.)
is available under pymrsf.experimental as a research-grade backend.
"""

import logging
import os
from dataclasses import dataclass, field

from . import experimental

# Experimental research backend — re-exported at top level for backward compat
from .experimental import (
    close_connections,
    load_index,
    mrsf_benchmark_canterbury,
    mrsf_delete,
    mrsf_inspect,
    mrsf_latency_benchmark,
    mrsf_read,
    mrsf_read_novel,
    mrsf_rebuild_explained,
    mrsf_write,
    rebuild_index,
    reset_index_metadata,
    save_index,
)


def rebuild_faiss_from_sqlite(*args, **kwargs):
    """Deprecated alias for reset_index_metadata(). Removed in v0.6."""
    import warnings
    warnings.warn(
        "rebuild_faiss_from_sqlite is deprecated and will be removed in v0.6. "
        "Use pymrsf.reset_index_metadata() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return reset_index_metadata(*args, **kwargs)
from . import cache
from .cache import (
    clear_cache,
    clear_embedding_cache,
    configure_cache,
    get_cache_stats,
    get_embedding_cache_stats,
    reset_cache_stats,
)
from .chunker import smart_chunk
from .core import (
    LOGIT_PRECISION,
    MODEL_VERSION,
    PROVIDER,
    ModelSession,
    compute_delta,
    detokenize,
    get_backend,
    get_raw_lm,
    get_surprises,
    next_token_greedy,
    provider_capabilities,
    quantized_argmax,
    set_provider,
    tokenize,
)
from .embeddings import embed, get_embedding_dim
from .probe import probe, probe_compare
from .rag import (
    DEFAULT_RAG_THRESHOLDS,
    DEFAULT_RELEVANCE_CUTOFF,
    DEFAULT_WEIGHTS,
    WeightConfig,
    explain_chunk,
    filter_chunks,
    filter_chunks_async,
    score_chunk,
    score_chunk_async,
    score_chunks,
    score_chunks_async,
    score_chunks_batch,
    smart_filter,
)

__version__ = "0.6.0"


# ── Centralized runtime configuration ─────────────────────────────────────────

@dataclass
class Config:
    """Central configuration for all pymrsf settings.

    Reads defaults from environment variables so existing .env setups
    continue to work. Override at runtime with configure().

    Live-reconfigurable fields (take effect on next call):
        provider, ollama_base, embed_model, embed_timeout,
        default_relevance_cutoff, default_min_rag_score, default_diversity_threshold

    Import-time only fields (model already loaded; changing has no effect):
        model_path, n_ctx, n_gpu_layers, logit_precision
    """
    provider: str = field(default_factory=lambda: os.getenv("PYMRSF_PROVIDER", "local"))
    model_path: str = field(
        default_factory=lambda: os.getenv("PYMRSF_MODEL_PATH", "./models/mistral-7b-v0.1.Q4_K_M.gguf")
    )
    n_ctx: int = field(default_factory=lambda: int(os.getenv("PYMRSF_N_CTX", "4096")))
    n_gpu_layers: int = field(default_factory=lambda: int(os.getenv("PYMRSF_N_GPU_LAYERS", "0")))
    ollama_base: str = field(default_factory=lambda: os.getenv("PYMRSF_OLLAMA_BASE", "http://localhost:11434"))
    embed_model: str = field(default_factory=lambda: os.getenv("PYMRSF_EMBED_MODEL", "nomic-embed-text"))
    embed_timeout: int = field(default_factory=lambda: int(os.getenv("PYMRSF_EMBED_TIMEOUT", "30")))
    embed_dim: int = field(default_factory=lambda: int(os.getenv("PYMRSF_EMBED_DIM", "768")))
    logit_precision: int = field(default_factory=lambda: int(os.getenv("PYMRSF_LOGIT_PRECISION", "6")))
    db_path: str = field(default_factory=lambda: os.getenv("PYMRSF_DB_PATH", "mrsf.db"))
    faiss_path: str = field(default_factory=lambda: os.getenv("PYMRSF_FAISS_PATH", "mrsf.faiss"))
    default_relevance_cutoff: float = 0.30
    default_min_rag_score: int = 50
    default_diversity_threshold: float = 0.85


_config = Config()


def configure_logging(level: str = "INFO") -> None:
    """Configure the pymrsf logger.

    Call this once at application startup to see library log output.
    By default pymrsf ships with a NullHandler so it never pollutes stdout
    unless the application explicitly enables logging.

    Args:
        level: Log level string — "DEBUG", "INFO", "WARNING", "ERROR".

    Example:
        import pymrsf
        pymrsf.configure_logging("DEBUG")   # show all internal messages
        pymrsf.configure_logging("WARNING") # only warnings and errors
    """
    pkg_logger = logging.getLogger("pymrsf")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def configure(**kwargs) -> Config:
    """Set global pymrsf configuration at runtime.

    Accepts any field name from Config. Individual modules continue
    to read os.getenv for now (backward compat with existing .env setups).

    Example:
        import pymrsf
        pymrsf.configure(provider="openai", default_relevance_cutoff=0.40)
    """
    global _config
    for k, v in kwargs.items():
        if hasattr(_config, k):
            setattr(_config, k, v)
        else:
            raise ValueError(f"Unknown config key: '{k}'. Valid keys: {list(_config.__dataclass_fields__)}")
    return _config


def get_config() -> Config:
    """Return the current global Config instance."""
    return _config

# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    # ── RAG scoring (headline API) ─────────────────────────────────────────────
    "score_chunk",
    "score_chunks",
    "score_chunks_batch",
    "explain_chunk",
    "filter_chunks",
    "smart_filter",
    "DEFAULT_WEIGHTS",
    "DEFAULT_RELEVANCE_CUTOFF",
    "DEFAULT_RAG_THRESHOLDS",
    "WeightConfig",

    # Async variants
    "score_chunk_async",
    "score_chunks_async",
    "filter_chunks_async",

    # Chunking
    "smart_chunk",

    # Knowledge probing
    "probe",
    "probe_compare",

    # Embeddings
    "embed",
    "get_embedding_dim",

    # ── Core / backend ─────────────────────────────────────────────────────────
    "tokenize",
    "detokenize",
    "quantized_argmax",
    "get_surprises",
    "compute_delta",
    "next_token_greedy",
    "ModelSession",
    "get_backend",
    "get_raw_lm",
    "provider_capabilities",
    "set_provider",
    "PROVIDER",
    "MODEL_VERSION",
    "LOGIT_PRECISION",

    # ── Cache ──────────────────────────────────────────────────────────────────
    "cache",
    "configure_cache",
    "get_cache_stats",
    "get_embedding_cache_stats",
    "reset_cache_stats",
    "clear_cache",
    "clear_embedding_cache",

    # ── Runtime configuration ──────────────────────────────────────────────────
    "Config",
    "configure",
    "get_config",
    "configure_logging",

    # ── Experimental: MRSF storage backend ────────────────────────────────────
    "experimental",
    "mrsf_write",
    "mrsf_read",
    "mrsf_read_novel",
    "mrsf_delete",
    "rebuild_index",
    "save_index",
    "load_index",
    "reset_index_metadata",
    "rebuild_faiss_from_sqlite",  # Deprecated alias — removed in v0.6
    "close_connections",
    "mrsf_inspect",
    "mrsf_rebuild_explained",
    "mrsf_benchmark_canterbury",
    "mrsf_latency_benchmark",

    # Version
    "__version__",
]
