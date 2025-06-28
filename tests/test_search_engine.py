import pytest
pytest.importorskip("numpy")

from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shared.search_engine as knowledge_search


def test_hybrid_search_output_format(monkeypatch):
    engine = knowledge_search.HybridSearchEngine.__new__(knowledge_search.HybridSearchEngine)
    engine.chunks = [{"id": "c1", "text": "hello world", "metadata": {}}]
    engine.embeddings = {"c1": [1.0]}
    engine.bm25_index = SimpleNamespace(get_scores=lambda tokens: [1.0])
    engine.tokenized_corpus_for_bm25 = [["hello"]]
    engine.model = None
    engine.embedding_model = "dummy"

    monkeypatch.setattr(engine, "get_embedding_from_openai", lambda q, client=None: [1.0])
    monkeypatch.setattr(knowledge_search, "tokenize_text_for_bm25_internal", lambda q: ["hello"])

    results, not_found = engine.search("hello", top_k=1, threshold=0.0, vector_weight=1.0, bm25_weight=1.0)
    assert not not_found
    assert isinstance(results, list) and results
    first = results[0]
    assert {"id", "text", "metadata", "similarity", "vector_score", "bm25_score"} <= first.keys()
