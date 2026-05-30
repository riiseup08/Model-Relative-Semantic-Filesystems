"""Tests for async scoring functions (score_chunk_async, score_chunks_async, filter_chunks_async)."""
import asyncio
import pytest
from unittest.mock import patch, MagicMock


FAKE_PROBE = {"knowledge_score": 40, "token_count": 10, "surprise_count": 3,
              "compression": 0.7, "label": "uncommon", "description": "...",
              "surprises": [], "heatmap": [], "model": "test"}
FAKE_EMBED = [0.1] * 768


def _make_caps(probe=False):
    return {"supports_probe": probe, "supports_embeddings": True,
            "supports_delta": probe, "provider": "local" if probe else "openai"}


@pytest.mark.asyncio
async def test_basic_async_score():
    with patch("pymrsf.rag.provider_capabilities", return_value=_make_caps()), \
         patch("pymrsf.rag.embed", return_value=FAKE_EMBED), \
         patch("pymrsf.rag.probe", None):
        from pymrsf.rag import score_chunk_async
        result = await score_chunk_async("some text chunk", query="what is this?")
    assert "rag_score" in result
    assert 0 <= result["rag_score"] <= 100
    assert "verdict" in result


@pytest.mark.asyncio
async def test_async_matches_sync():
    """Async and sync should produce identical results when given same inputs."""
    with patch("pymrsf.rag.provider_capabilities", return_value=_make_caps()), \
         patch("pymrsf.rag.embed", return_value=FAKE_EMBED), \
         patch("pymrsf.rag.probe", None):
        from pymrsf.rag import score_chunk, score_chunk_async
        sync_result = score_chunk("test chunk", query="test query", use_cache=False)
        async_result = await score_chunk_async("test chunk", query="test query", use_cache=False)
    assert sync_result["rag_score"] == async_result["rag_score"]
    assert sync_result["verdict"] == async_result["verdict"]


@pytest.mark.asyncio
async def test_score_chunks_async_returns_ranked():
    chunks = ["chunk one", "chunk two", "chunk three"]
    with patch("pymrsf.rag.provider_capabilities", return_value=_make_caps()), \
         patch("pymrsf.rag.embed", return_value=FAKE_EMBED), \
         patch("pymrsf.rag.probe", None):
        from pymrsf.rag import score_chunks_async
        results = await score_chunks_async(chunks, query="test query")
    assert len(results) == 3
    # Results should be sorted best-first
    scores = [r["rag_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # Each result should have a rank
    assert all("rank" in r for r in results)


@pytest.mark.asyncio
async def test_score_chunks_async_empty():
    from pymrsf.rag import score_chunks_async
    results = await score_chunks_async([])
    assert results == []


@pytest.mark.asyncio
async def test_incremental_async_runs_without_error():
    """incremental=True with probe unavailable should fall back gracefully."""
    with patch("pymrsf.rag.provider_capabilities", return_value=_make_caps(probe=False)), \
         patch("pymrsf.rag.embed", return_value=FAKE_EMBED), \
         patch("pymrsf.rag.probe", None):
        from pymrsf.rag import score_chunks_async
        results = await score_chunks_async(
            ["chunk a", "chunk b"], query="query", incremental=True
        )
    assert len(results) == 2


@pytest.mark.asyncio
async def test_diversity_dedup_async():
    """Two identical chunks should result in the second being marked as duplicate."""
    identical = "This is the same text repeated exactly."
    with patch("pymrsf.rag.provider_capabilities", return_value=_make_caps()), \
         patch("pymrsf.rag.embed", return_value=FAKE_EMBED), \
         patch("pymrsf.rag.probe", None):
        from pymrsf.rag import score_chunks_async
        results = await score_chunks_async(
            [identical, identical], query="something", diversity_threshold=0.85
        )
    # At least one should be marked as skip/0 due to dedup
    verdicts = {r["verdict"] for r in results}
    assert "skip" in verdicts or any(r["rag_score"] == 0 for r in results)


@pytest.mark.asyncio
async def test_filter_chunks_async_returns_strings():
    chunks = ["useful chunk", "another chunk"]
    with patch("pymrsf.rag.provider_capabilities", return_value=_make_caps()), \
         patch("pymrsf.rag.embed", return_value=FAKE_EMBED), \
         patch("pymrsf.rag.probe", None):
        from pymrsf.rag import filter_chunks_async
        result = await filter_chunks_async(chunks, query="query", min_rag_score=0)
    assert isinstance(result, list)
    assert all(isinstance(c, str) for c in result)


@pytest.mark.asyncio
async def test_filter_chunks_async_respects_min_score():
    chunks = ["chunk a", "chunk b"]
    with patch("pymrsf.rag.provider_capabilities", return_value=_make_caps()), \
         patch("pymrsf.rag.embed", return_value=FAKE_EMBED), \
         patch("pymrsf.rag.probe", None):
        from pymrsf.rag import filter_chunks_async
        # min_rag_score=101 means nothing passes
        result = await filter_chunks_async(chunks, query="query", min_rag_score=101)
    assert result == []


@pytest.mark.asyncio
async def test_custom_thresholds_async():
    """Passing custom thresholds should affect the verdict in async path."""
    tiny_thresholds = [(50, "high", "High"), (0, "low", "Low")]
    with patch("pymrsf.rag.provider_capabilities", return_value=_make_caps()), \
         patch("pymrsf.rag.embed", return_value=FAKE_EMBED), \
         patch("pymrsf.rag.probe", None):
        from pymrsf.rag import score_chunk_async
        result = await score_chunk_async(
            "text", query="q", thresholds=tiny_thresholds, use_cache=False
        )
    assert result["verdict"] in {"high", "low"}
