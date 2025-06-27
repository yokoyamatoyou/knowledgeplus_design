import pytest
pytest.importorskip("PyPDF2")

from io import BytesIO
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import upload_utils
from shared.kb_builder import KnowledgeBuilder


def test_build_from_file_creates_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)

    builder = KnowledgeBuilder()
    builder.get_openai_client = lambda: object()
    builder.get_embedding = lambda text, client=None: [0.1]
    builder.refresh_search_engine = lambda name: None

    buf = BytesIO(b"hello world")
    buf.name = "sample.txt"

    result = builder.build_from_file(buf)
    assert result is not None

    chunk_path = Path(result["chunk_path"])
    emb_path = Path(result["embedding_path"])
    meta_path = Path(result["metadata_path"])
    assert chunk_path.exists()
    assert emb_path.exists()
    assert meta_path.exists()
