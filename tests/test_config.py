"""Tests for Task 3.3: configure() actually configures live-reconfigurable fields."""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

import pymrsf


def _reset_config():
    pymrsf._config = pymrsf.Config()


def test_configure_returns_config():
    _reset_config()
    cfg = pymrsf.configure(embed_timeout=99)
    assert cfg.embed_timeout == 99
    _reset_config()


def test_configure_unknown_key_raises():
    _reset_config()
    with pytest.raises(ValueError, match="Unknown config key"):
        pymrsf.configure(nonexistent_key="oops")


def test_get_config_reflects_changes():
    _reset_config()
    pymrsf.configure(default_relevance_cutoff=0.55)
    assert pymrsf.get_config().default_relevance_cutoff == 0.55
    _reset_config()


def test_embed_timeout_is_live():
    """configure(embed_timeout=...) must be picked up by the next embed call."""
    _reset_config()
    pymrsf.configure(embed_timeout=77)

    captured = {}

    def fake_post(url, json, timeout):
        captured["timeout"] = timeout
        resp = MagicMock()
        resp.json.return_value = {"embeddings": [[0.1] * 768]}
        resp.raise_for_status = lambda: None
        return resp

    with patch("requests.post", side_effect=fake_post):
        try:
            pymrsf.embed("hello")
        except Exception:
            pass  # we only care that the timeout was passed correctly

    assert captured.get("timeout") == 77
    _reset_config()


def test_embed_model_is_live():
    """configure(embed_model=...) must be picked up by the next embed call."""
    _reset_config()
    pymrsf.configure(embed_model="mxbai-embed-large")

    captured = {}

    def fake_post(url, json, timeout):
        captured["model"] = json.get("model")
        resp = MagicMock()
        resp.json.return_value = {"embeddings": [[0.2] * 768]}
        resp.raise_for_status = lambda: None
        return resp

    with patch("requests.post", side_effect=fake_post):
        try:
            pymrsf.embed("hello")
        except Exception:
            pass

    assert captured.get("model") == "mxbai-embed-large"
    _reset_config()
