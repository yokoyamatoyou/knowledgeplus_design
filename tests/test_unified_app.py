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
