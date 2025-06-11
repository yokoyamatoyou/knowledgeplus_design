from __future__ import annotations

import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from knowledge_gpt_app.knowledge_search import HybridSearchEngine
from .upload_utils import BASE_KNOWLEDGE_DIR


_search_engines: Dict[str, HybridSearchEngine] = {}


def get_search_engine(kb_name: str) -> Optional[HybridSearchEngine]:
    """Return a cached search engine for the given knowledge base."""
    kb_dir = BASE_KNOWLEDGE_DIR / kb_name
    if not kb_dir.exists():
        return None
    engine = _search_engines.get(kb_name)
    if engine is None:
        try:
            engine = HybridSearchEngine(str(kb_dir))
            _search_engines[kb_name] = engine
        except Exception:
            return None
    return engine


def refresh_search_engine(kb_name: str) -> None:
    engine = _search_engines.get(kb_name)
    if engine:
        try:
            engine.reindex()
        except Exception:
            pass


def search_kb(
    kb_name: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.15,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    client=None,
) -> Tuple[List[dict], bool]:
    """Execute a hybrid search for a single knowledge base."""
    engine = get_search_engine(kb_name)
    if engine is None:
        return [], True
    return engine.search(
        query,
        top_k=top_k,
        threshold=threshold,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        client=client,
    )
