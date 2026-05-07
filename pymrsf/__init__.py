from .storage import mrsf_write, mrsf_read, save_index, load_index
from .inspect import mrsf_inspect, mrsf_rebuild_explained
from .benchmark import mrsf_benchmark_canterbury, mrsf_latency_benchmark
from .embeddings import embed
from .probe import probe, probe_compare
from .rag import (
    score_chunk, score_chunks, score_chunks_batch,
    explain_chunk, filter_chunks, DEFAULT_WEIGHTS,
)
from .core import ModelSession, compute_delta, get_surprises, tokenize, detokenize

__version__ = "0.4.0"
