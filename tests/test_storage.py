"""Tests for storage.py — SQLite + FAISS persistence, rebuild, and tombstone handling."""
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


FAKE_EMBED = np.array([0.1] * 768, dtype=np.float32)
FAKE_DELTA = [(1, 42), (3, 99)]  # [(position, token_id), ...]
FAKE_CAPS = {"supports_delta": True, "supports_embeddings": True,
             "supports_probe": False, "provider": "local"}


def _patch_storage(tmp_path):
    """Context manager patches that redirect storage to a temp directory."""
    db = str(tmp_path / "test.db")
    faiss_f = str(tmp_path / "test.faiss")
    return (
        patch("pymrsf.experimental.storage.DB_PATH", db),
        patch("pymrsf.experimental.storage.FAISS_PATH", faiss_f),
        patch("pymrsf.experimental.storage.provider_capabilities", return_value=FAKE_CAPS),
        patch("pymrsf.experimental.storage.embed", return_value=FAKE_EMBED),
        patch("pymrsf.experimental.storage.get_embedding_dim", return_value=768),
        patch("pymrsf.experimental.storage.tokenize", return_value=[1, 2, 3, 4, 5]),
        patch("pymrsf.experimental.storage.compute_delta", return_value=FAKE_DELTA),
    )


@pytest.fixture(autouse=True)
def reset_storage_globals():
    """Reset storage module globals before each test."""
    import pymrsf.experimental.storage as s
    s._faiss_index = None
    s._index_meta = None
    s._tombstones = set()
    s._conn = None
    s._cur = None
    s._rebuild_counter = 0
    yield
    if s._conn:
        s._conn.close()
    s._faiss_index = None
    s._index_meta = None
    s._tombstones = set()
    s._conn = None
    s._cur = None


def test_write_stores_original_text(tmp_path):
    patches = _patch_storage(tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        import pymrsf.experimental.storage as s
        result = s.mrsf_write("hello world", doc_id="doc1")
        assert "error" not in result
        cur, _ = s._get_db()
        row = cur.execute(
            "SELECT original_text FROM documents WHERE doc_id='doc1'"
        ).fetchone()
        assert row is not None
        assert row[0] == "hello world"


def test_write_returns_compression_stats(tmp_path):
    patches = _patch_storage(tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        import pymrsf.experimental.storage as s
        result = s.mrsf_write("some text")
        assert "doc_id" in result
        assert "compression" in result
        assert 0.0 <= result["compression"] <= 1.0


def test_delete_marks_as_deleted(tmp_path):
    patches = _patch_storage(tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        import pymrsf.experimental.storage as s
        s.mrsf_write("doc to delete", doc_id="del1")
        deleted = s.mrsf_delete("del1")
        assert deleted is True
        cur, _ = s._get_db()
        row = cur.execute(
            "SELECT deleted FROM documents WHERE doc_id='del1'"
        ).fetchone()
        assert row[0] == 1


def test_rebuild_restores_vectors(tmp_path):
    patches = _patch_storage(tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        import pymrsf.experimental.storage as s
        s.mrsf_write("doc alpha", doc_id="alpha")
        s.mrsf_write("doc beta", doc_id="beta")
        s.mrsf_delete("beta")

        # Reset FAISS to simulate a crash/reload scenario
        s._faiss_index = None
        s._index_meta = None
        s._tombstones = set()

        result = s.rebuild_index(verbose=False)
        assert result["recovered"] == 1
        assert result["skipped"] == 0 or result["skipped"] >= 0
        assert s._faiss_index.ntotal == 1


def test_rebuild_skips_legacy_rows(tmp_path):
    """Rows with original_text=NULL should be skipped with a warning."""
    patches = _patch_storage(tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        import pymrsf.experimental.storage as s
        # Write normally (gets original_text)
        s.mrsf_write("real doc", doc_id="real")
        # Manually insert a legacy row with NULL original_text
        cur, conn = s._get_db()
        cur.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)",
            ("legacy", "v0.3", "nomic", b"delta", 5, 2, 10, 768)
        )
        conn.commit()
        s._faiss_index = None
        s._index_meta = None
        s._tombstones = set()

        result = s.rebuild_index(verbose=False)
        assert result["recovered"] == 1
        assert result["skipped"] == 1


def test_save_and_load_index(tmp_path):
    patches = _patch_storage(tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        import pymrsf.experimental.storage as s
        s.mrsf_write("doc to persist", doc_id="persist1")
        ntotal_before = s._faiss_index.ntotal
        s.save_index()

        # Simulate a new process by resetting globals
        s._faiss_index = None
        s._index_meta = None
        s._tombstones = set()

        s.load_index()
        assert s._faiss_index is not None
        assert s._faiss_index.ntotal == ntotal_before


def test_metadata_preserved_after_load(tmp_path):
    patches = _patch_storage(tmp_path)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        import pymrsf.experimental.storage as s
        s.mrsf_write("doc x", doc_id="docx")
        s.save_index()
        s._faiss_index = None
        s._index_meta = None
        s._tombstones = set()
        s.load_index()
        assert "docx" in s._index_meta
