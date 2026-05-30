"""Integration tests — end-to-end RAG pipeline scenarios."""
import pytest
from unittest.mock import patch


FAKE_EMBED = [0.1] * 768
FAKE_CAPS_API = {"supports_probe": False, "supports_embeddings": True,
                 "supports_delta": False, "provider": "openai"}


def _patch_api():
    return (
        patch("pymrsf.rag.provider_capabilities", return_value=FAKE_CAPS_API),
        patch("pymrsf.rag.embed", return_value=FAKE_EMBED),
        patch("pymrsf.rag.probe", None),
    )


# ── Threshold tests ────────────────────────────────────────────────────────────

def test_configurable_relevance_cutoff_changes_score():
    """A higher relevance_cutoff should lower or zero relevance_score."""
    with _patch_api()[0], _patch_api()[1], _patch_api()[2]:
        from pymrsf.rag import score_chunk
        r_low  = score_chunk("text", query="q", relevance_cutoff=0.0,  use_cache=False)
        r_high = score_chunk("text", query="q", relevance_cutoff=0.999, use_cache=False)
    assert r_low["relevance_score"] >= r_high["relevance_score"]


def test_custom_thresholds_change_verdict():
    """Custom thresholds should override the default verdict labels."""
    custom = [(50, "great", "Great chunk"), (0, "meh", "Meh chunk")]
    with _patch_api()[0], _patch_api()[1], _patch_api()[2]:
        from pymrsf.rag import score_chunk
        result = score_chunk("some chunk", thresholds=custom, use_cache=False)
    assert result["verdict"] in {"great", "meh"}


def test_filter_chunks_custom_thresholds():
    """Custom thresholds should propagate through filter_chunks."""
    chunks = ["chunk a", "chunk b", "chunk c"]
    custom = [(0, "ok", "ok")]  # Every score passes at 0
    with _patch_api()[0], _patch_api()[1], _patch_api()[2]:
        from pymrsf.rag import filter_chunks
        result = filter_chunks(chunks, query="q", min_rag_score=0, thresholds=custom)
    assert isinstance(result, list)
    assert len(result) == len(chunks)


# ── Weight redistribution tests ────────────────────────────────────────────────

def test_query_ignorance_weight_redistributed_when_no_probe():
    """When probe is unavailable, query_ignorance weight should shift to novelty/relevance."""
    with _patch_api()[0], _patch_api()[1], _patch_api()[2]:
        from pymrsf.rag import score_chunk
        result = score_chunk(
            "text", query="q",
            weights={"novelty": 0.3, "relevance": 0.3, "query_ignorance": 0.4},
            use_cache=False,
        )
    # weights_used should have query_ignorance=0 (redistributed)
    assert result["weights_used"]["query_ignorance"] == 0.0
    assert abs(result["weights_used"]["novelty"] + result["weights_used"]["relevance"] - 1.0) < 0.01


# ── WeightConfig tests ─────────────────────────────────────────────────────────

def test_weightconfig_normalize():
    from pymrsf.rag import WeightConfig
    wc = WeightConfig(novelty=1.0, relevance=1.0, query_ignorance=2.0).normalize()
    assert abs(wc.novelty + wc.relevance + wc.query_ignorance - 1.0) < 1e-9


def test_weightconfig_redistribute():
    from pymrsf.rag import WeightConfig
    wc = WeightConfig(novelty=0.4, relevance=0.4, query_ignorance=0.2)
    wc2 = wc.redistribute_for_relevance_only()
    assert wc2.query_ignorance == 0.0
    assert abs(wc2.novelty + wc2.relevance - 1.0) < 1e-9


def test_weightconfig_from_dict_fills_missing_keys():
    from pymrsf.rag import WeightConfig
    wc = WeightConfig.from_dict({"novelty": 1.0})
    assert wc.relevance > 0
    assert wc.query_ignorance > 0


# ── Default constants backward compat ─────────────────────────────────────────

def test_default_constants_exported():
    from pymrsf import DEFAULT_RELEVANCE_CUTOFF, DEFAULT_RAG_THRESHOLDS, DEFAULT_WEIGHTS, WeightConfig
    assert isinstance(DEFAULT_RELEVANCE_CUTOFF, float)
    assert isinstance(DEFAULT_RAG_THRESHOLDS, list)
    assert len(DEFAULT_RAG_THRESHOLDS) > 0
    assert isinstance(DEFAULT_WEIGHTS, dict)
    assert WeightConfig is not None


def test_rag_thresholds_backward_compat_alias():
    """RAG_THRESHOLDS should still be importable (backward compat)."""
    from pymrsf.rag import RAG_THRESHOLDS, DEFAULT_RAG_THRESHOLDS
    assert RAG_THRESHOLDS is DEFAULT_RAG_THRESHOLDS


# ── Pipeline smoke test ────────────────────────────────────────────────────────

def test_score_chunks_pipeline():
    chunks = ["Neural networks learn via gradient descent.",
              "Paris is the capital of France.",
              "Transformers use self-attention mechanisms."]
    with _patch_api()[0], _patch_api()[1], _patch_api()[2]:
        from pymrsf.rag import score_chunks
        results = score_chunks(chunks, query="how do transformers learn?")
    assert len(results) == len(chunks)
    assert all("rag_score" in r for r in results)
    assert all("rank" in r for r in results)
    # Ranks should be 1, 2, 3
    assert sorted(r["rank"] for r in results) == [1, 2, 3]
