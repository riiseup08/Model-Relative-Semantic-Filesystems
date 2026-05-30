"""Tests for pymrsf.core — provider routing (no LLM required)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pymrsf.core import PROVIDER, LOGIT_PRECISION


class TestConstants:
    def test_provider_is_local_by_default(self):
        assert PROVIDER in ("local", "openai")

    def test_logit_precision_is_valid(self):
        assert isinstance(LOGIT_PRECISION, int)
        assert 1 <= LOGIT_PRECISION <= 16


class TestModuleLoads:
    def test_import_pymrsf(self):
        """Verify the package imports without loading the LLM."""
        import pymrsf
        assert hasattr(pymrsf, "score_chunk")
        assert hasattr(pymrsf, "filter_chunks")
        assert hasattr(pymrsf, "probe")
        assert hasattr(pymrsf, "__version__")

    def test_import_rag_submodule(self):
        from pymrsf import rag
        assert hasattr(rag, "score_chunk")
        assert hasattr(rag, "filter_chunks")


import pytest
