"""Local-provider regression tests (require a real GGUF model).

Skipped automatically when the model file is absent or llama-cpp-python isn't
installed, so the suite stays green on machines without a local model. These
exercise the parts that can't be tested with the API providers:

  #3   concurrency safety on the single shared model (deterministic results)
  #11  smooth probability signal (confidence/perplexity) + default unchanged
  parity between sync and batch scoring paths

To keep them fast and Ollama-free, scoring tests use query=None (novelty-only),
which exercises probe() without needing the embedding endpoint.
"""
import asyncio
import os

import pytest

from conftest import local_model_path

pytestmark = pytest.mark.skipif(
    not os.path.exists(local_model_path()),
    reason=f"local GGUF not found at {local_model_path()} (set PYMRSF_TEST_MODEL)",
)

CHUNKS = [
    "Photosynthesis converts light into chemical energy in plants.",
    "The mitochondria is the powerhouse of the cell.",
    "Neural networks are trained with gradient descent.",
    "The French Revolution began in 1789.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Quantum entanglement links particle states across distance.",
]


@pytest.fixture(scope="module")
def local():
    """Configure pymrsf for the local provider and load the model once."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        pytest.skip("llama-cpp-python not installed")

    os.environ["PYMRSF_MODEL_PATH"] = local_model_path()
    os.environ["PYMRSF_N_CTX"] = "2048"
    os.environ.setdefault("PYMRSF_MODEL_VERSION", "test-local")

    import pymrsf
    pymrsf.set_provider("local")
    assert pymrsf.provider_capabilities()["supports_probe"] is True
    yield pymrsf


# ── #11 smooth signal, default unchanged ────────────────────────────────────
def test_probe_default_metric_is_argmax_and_discriminates(local):
    famous = local.probe("The quick brown fox jumps over the lazy dog.")
    novel = local.probe("Zxqv blorptang mernquist 88 flooble wibbenshaw.")
    for r in (famous, novel):
        assert "error" not in r
        assert r["metric"] == "argmax"
        # default knowledge_score must equal the argmax compression metric
        assert r["knowledge_score"] == round(r["compression"] * 100)
        # new smooth fields present and well-formed
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["perplexity"] >= 1.0
        assert all(0.0 <= h["prob"] <= 1.0 for h in r["heatmap"])
    # signal discriminates known from gibberish on both metrics
    assert famous["knowledge_score"] > novel["knowledge_score"]
    assert famous["confidence"] > novel["confidence"]
    assert famous["perplexity"] < novel["perplexity"]


def test_confidence_metric_is_opt_in(local, monkeypatch):
    monkeypatch.setenv("PYMRSF_PROBE_METRIC", "confidence")
    r = local.probe("The quick brown fox jumps over the lazy dog.")
    assert r["metric"] == "confidence"
    assert r["knowledge_score"] == round(r["confidence"] * 100)


# ── #3 concurrency safety ───────────────────────────────────────────────────
def test_concurrent_async_scoring_is_deterministic(local):
    """Parallel scoring on the shared model must not corrupt KV state."""
    async def run():
        a = await local.score_chunks_async(list(CHUNKS), query=None, max_concurrent=4,
                                            diversity_threshold=1.0)
        b = await local.score_chunks_async(list(CHUNKS), query=None, max_concurrent=4,
                                            diversity_threshold=1.0)
        return a, b

    a, b = asyncio.run(run())
    sa = {r["chunk"]: r["rag_score"] for r in a}
    sb = {r["chunk"]: r["rag_score"] for r in b}
    assert sa == sb, "concurrent runs diverged -> model_lock failed"


# ── sync/batch parity (#9/#10 behaviour-preserving) ─────────────────────────
def test_sync_and_batch_scores_match(local):
    sync = {r["chunk"]: r["rag_score"]
            for r in local.score_chunks(list(CHUNKS), query=None, diversity_threshold=1.0)}
    batch = {r["chunk"]: r["rag_score"]
             for r in local.score_chunks_batch(list(CHUNKS), query=None, diversity_threshold=1.0)}
    assert sync == batch
