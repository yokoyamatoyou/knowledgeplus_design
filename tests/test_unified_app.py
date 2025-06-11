import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_sidebar_has_faq_button():
    text = Path('unified_app.py').read_text(encoding='utf-8')
    assert 'FAQ生成' in text
    assert '処理モード' in text
    assert 'インデックス更新' in text
    assert '検索インデックス更新' in text
