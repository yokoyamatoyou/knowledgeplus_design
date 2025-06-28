import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DEFAULT_KB_NAME

def test_sidebar_has_faq_button():
    text = Path('unified_app.py').read_text(encoding='utf-8')
    assert 'FAQ生成' in text
    assert '処理モード' in text
    assert '個別処理' in text
    assert 'まとめて処理' in text
    assert 'インデックス更新' in text
    assert '自動(処理後)' in text
    assert '手動' in text
    assert '検索インデックス更新' in text


def test_manual_refresh_call_present():
    text = Path('unified_app.py').read_text(encoding='utf-8')
    import re
    pattern = r'if st\.button\("検索インデックス更新"\).*refresh_search_engine\(DEFAULT_KB_NAME\)'
    assert re.search(pattern, text, re.DOTALL)


def test_safe_generate_handles_error(monkeypatch):
    import types
    pytest = __import__('pytest')
    pytest.importorskip('streamlit')

    import streamlit as st
    monkeypatch.setattr(st, 'error', lambda msg: monkeypatch.setattr(st, '_err', msg))

    import importlib
    monkeypatch.setattr('ui_modules.theme.apply_intel_theme', lambda *a, **k: None)
    monkeypatch.setattr(st, 'set_page_config', lambda *a, **k: None)
    monkeypatch.setattr(st, 'title', lambda *a, **k: None)
    sidebar = types.SimpleNamespace(radio=lambda *a, **k: 'FAQ')
    monkeypatch.setattr(st, 'sidebar', sidebar)
    monkeypatch.setattr(st, 'info', lambda *a, **k: None)

    mod = importlib.reload(__import__('unified_app'))

    def boom(*a, **k):
        raise RuntimeError('fail')

    monkeypatch.setattr(mod, 'generate_gpt_response', boom)
    result = mod.safe_generate_gpt_response('prompt', conversation_history=[], persona='default', temperature=0.1, response_length='簡潔', client=None)
    assert result is None
    assert getattr(st, '_err', None)
