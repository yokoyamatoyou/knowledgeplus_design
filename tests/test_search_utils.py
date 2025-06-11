import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
pytest.importorskip("numpy")
pytest.importorskip("rank_bm25")
pytest.importorskip("sentence_transformers")

from shared import upload_utils, search_utils


def test_get_search_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(search_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    kb = "kb"
    upload_utils.save_processed_data(kb, "1", chunk_text="a", embedding=[1], metadata={})
    engine = search_utils.get_search_engine(kb)
    assert engine is not None
    assert kb in search_utils._search_engines
