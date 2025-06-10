
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Provide stub for openai so faq_utils can be imported without the real package
sys.modules['openai'] = types.SimpleNamespace(OpenAI=lambda **_: None)

from shared import faq_utils


class DummyClient:
    def __init__(self, content):
        self._content = content
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, outer):
            self.completions = outer.Completions(outer)

    class Completions:
        def __init__(self, outer):
            self._content = outer._content

        def create(self, *args, **kwargs):
            return type(
                "Resp",
                (),
                {
                    "choices": [
                        type("Choice", (), {"message": type("Msg", (), {"content": self._content})()})
                    ]
                },
            )()


def test_generate_faqs_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(faq_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    kb_dir = tmp_path / "kb"
    chunk_dir = kb_dir / "chunks"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "1.txt").write_text("hello", encoding="utf-8")

    client = DummyClient('{"question": "Q1", "answer": "A1"}')
    paths = faq_utils.generate_faqs_from_chunks("kb", client=client)

    assert len(paths) == 1
    data = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    assert data["question"] == "Q1"
    assert data["answer"] == "A1"
    assert data["source_chunk"] == "1"


def test_generate_faqs_separate_kb(tmp_path, monkeypatch):
    monkeypatch.setattr(faq_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    for name in ["kb1", "kb2"]:
        d = tmp_path / name / "chunks"
        d.mkdir(parents=True)
        (d / "x.txt").write_text("text", encoding="utf-8")
    client = DummyClient('{"question": "Q", "answer": "A"}')
    faq_utils.generate_faqs_from_chunks("kb1", client=client)
    faq_utils.generate_faqs_from_chunks("kb2", client=client)
    assert (tmp_path / "kb1" / "faq").exists()
    assert (tmp_path / "kb2" / "faq").exists()

