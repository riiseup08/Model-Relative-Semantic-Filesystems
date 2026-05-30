"""Tests for pymrsf.rag — RAG chunk quality scorer (no LLM required)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch
import pytest

from pymrsf.rag import (
    _cosine_similarity, _verdict, score_chunk,
    score_chunks, score_chunks_batch, filter_chunks,
    DEFAULT_WEIGHTS,
)
import numpy as np


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        result = _cosine_similarity(a, b)
        assert isinstance(result, float)


class TestVerdict:
    def test_excellent(self):
        label, desc = _verdict(95)
        assert label == "excellent"

    def test_good(self):
        label, desc = _verdict(75)
        assert label == "good"

    def test_moderate(self):
        label, desc = _verdict(50)
        assert label == "moderate"

    def test_weak(self):
        label, desc = _verdict(30)
        assert label == "weak"

    def test_skip(self):
        label, desc = _verdict(5)
        assert label == "skip"

    def test_boundary_excellent(self):
        label, desc = _verdict(80)
        assert label == "excellent"

    def test_boundary_good(self):
        label, desc = _verdict(60)
        assert label == "good"


class TestDefaultWeights:
    def test_keys_exist(self):
        assert "novelty" in DEFAULT_WEIGHTS
        assert "relevance" in DEFAULT_WEIGHTS
        assert "query_ignorance" in DEFAULT_WEIGHTS

    def test_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_weights_updated(self):
        """Verify the new multi-factor formula: 40/40/20 instead of 60/40."""
        assert DEFAULT_WEIGHTS["novelty"] == 0.40
        assert DEFAULT_WEIGHTS["relevance"] == 0.40
        assert DEFAULT_WEIGHTS["query_ignorance"] == 0.20


class TestScoreChunkNoModel:
    @pytest.fixture(autouse=True)
    def _no_probe(self):
        with patch("pymrsf.rag._get_probe", return_value=None):
            yield

    def test_chunk_too_short(self):
        result = score_chunk("Hi", query="test")
        assert "error" in result or result["token_count"] <= 2

    def test_result_keys(self):
        """Verify the new result schema has query_knowledge and incremental_novelty."""
        result = score_chunk("Hi", query="test")
        if "error" not in result:
            assert "query_knowledge" in result
            assert "incremental_novelty" in result
            assert "query_ignorance" in result

    def test_custom_weights_passed(self):
        w = {"novelty": 0.5, "relevance": 0.3, "query_ignorance": 0.2}
        result = score_chunk("Hi", query="test", weights=w)
        # Should not crash
        assert isinstance(result, dict)


class TestEmptyInputs:
    @pytest.fixture(autouse=True)
    def _no_probe(self):
        with patch("pymrsf.rag._get_probe", return_value=None):
            yield

    def test_score_chunks_empty(self):
        assert score_chunks([], query="test") == []

    def test_score_chunks_batch_empty(self):
        assert score_chunks_batch([], query="test") == []

    def test_filter_chunks_empty(self):
        assert filter_chunks([], query="test") == []

    def test_filter_chunks_no_query(self):
        result = filter_chunks(["Hello world chunk test"], query="test", min_rag_score=0)
        assert isinstance(result, list)


class TestConditionalNoveltyProbeCount:
    """Verify compute_conditional_novelty flag controls the number of probe calls."""

    def _clear(self):
        from pymrsf import cache
        cache.clear_cache(reset_stats=True)
        cache.configure_cache(enabled=False)

    def _fake_probe(self, calls_list):
        def probe(text):
            calls_list.append(text)
            return {"knowledge_score": 50, "token_count": len(text.split()),
                    "surprise_count": 0}
        return probe

    def _caps(self):
        return {"supports_probe": True, "supports_embeddings": True,
                "supports_delta": True, "provider": "local"}

    def test_probe_count_off(self):
        """compute_conditional_novelty=False → N+1 probe calls (1 query + N chunks)."""
        self._clear()
        calls = []
        with patch("pymrsf.rag._get_probe", return_value=self._fake_probe(calls)), \
             patch("pymrsf.rag.provider_capabilities", return_value=self._caps()), \
             patch("pymrsf.rag.embed", return_value=[0.1] * 768):
            from pymrsf.rag import score_chunks
            score_chunks(["a", "b", "c"], query="test", compute_conditional_novelty=False)
        assert len(calls) == 4, f"Expected 4 probe calls, got {len(calls)}"

    def test_probe_count_on(self):
        """compute_conditional_novelty=True → 2N+1 probe calls (1 query + N chunks + N combined)."""
        self._clear()
        calls = []
        with patch("pymrsf.rag._get_probe", return_value=self._fake_probe(calls)), \
             patch("pymrsf.rag.provider_capabilities", return_value=self._caps()), \
             patch("pymrsf.rag.embed", return_value=[0.1] * 768):
            from pymrsf.rag import score_chunks
            score_chunks(["a", "b", "c"], query="test", compute_conditional_novelty=True)
        assert len(calls) == 7, f"Expected 7 probe calls, got {len(calls)}"

    def test_conditional_novelty_in_result(self):
        """When compute_conditional_novelty=True, conditional_novelty appears in result."""
        self._clear()
        calls = []
        with patch("pymrsf.rag._get_probe", return_value=self._fake_probe(calls)), \
             patch("pymrsf.rag.provider_capabilities", return_value=self._caps()), \
             patch("pymrsf.rag.embed", return_value=[0.1] * 768):
            from pymrsf.rag import score_chunks
            results = score_chunks(["chunk"], query="test", compute_conditional_novelty=True)
        assert "conditional_novelty" in results[0]
        assert results[0]["conditional_novelty"] >= 0

    def test_conditional_novelty_passes_through_filter_chunks(self):
        """filter_chunks should accept and forward compute_conditional_novelty."""
        self._clear()
        calls = []
        with patch("pymrsf.rag._get_probe", return_value=self._fake_probe(calls)), \
             patch("pymrsf.rag.provider_capabilities", return_value=self._caps()), \
             patch("pymrsf.rag.embed", return_value=[0.1] * 768):
            from pymrsf.rag import filter_chunks
            result = filter_chunks(["chunk"], query="test", compute_conditional_novelty=True)
        assert isinstance(result, list)

    def test_conditional_novelty_passes_through_smart_filter(self):
        """smart_filter should accept and forward compute_conditional_novelty."""
        self._clear()
        calls = []
        with patch("pymrsf.rag._get_probe", return_value=self._fake_probe(calls)), \
             patch("pymrsf.rag.provider_capabilities", return_value=self._caps()), \
             patch("pymrsf.rag.embed", return_value=[0.1] * 768):
            from pymrsf.rag import smart_filter
            result = smart_filter(["chunk"], query="test", compute_conditional_novelty=True)
        assert isinstance(result, dict)
        assert "chunks" in result

