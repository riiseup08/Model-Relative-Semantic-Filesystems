"""Tests for pymrsf.probe — knowledge probe logic (no LLM required)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pymrsf.probe import _label, THRESHOLDS


class TestLabel:
    def test_memorized(self):
        label, desc = _label(0.90)
        assert label == "memorized"

    def test_familiar(self):
        label, desc = _label(0.75)
        assert label == "familiar"

    def test_common(self):
        label, desc = _label(0.55)
        assert label == "common"

    def test_uncommon(self):
        label, desc = _label(0.35)
        assert label == "uncommon"

    def test_unknown(self):
        label, desc = _label(0.10)
        assert label == "unknown"

    def test_boundary_familiar(self):
        label, desc = _label(0.85)
        assert label == "memorized"

    def test_boundary_common(self):
        label, desc = _label(0.65)
        assert label == "familiar"

    def test_thresholds_ordered_descending(self):
        values = [t for t, _, _ in THRESHOLDS]
        assert all(values[i] >= values[i+1] for i in range(len(values) - 1))


import pytest
