import ast
from pathlib import Path


def _get_theme_call_args(path: str) -> int | None:
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == 'apply_intel_theme':
                return len(node.args)
    return None


def test_mm_kb_theme_call_uses_arg():
    assert _get_theme_call_args('mm_kb_builder/app.py') == 1


def test_gpt_app_theme_call_uses_arg():
    assert _get_theme_call_args('knowledge_gpt_app/app.py') == 1
