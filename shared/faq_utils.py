from __future__ import annotations
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from openai import OpenAI

from .upload_utils import BASE_KNOWLEDGE_DIR, ensure_openai_key


def _load_chunks(kb_name: str) -> Dict[str, str]:
    """Return mapping of chunk_id -> text for a knowledge base."""
    chunk_dir = BASE_KNOWLEDGE_DIR / kb_name / "chunks"
    texts: Dict[str, str] = {}
    if chunk_dir.exists():
        for p in chunk_dir.glob("*.txt"):
            texts[p.stem] = p.read_text(encoding="utf-8")
    return texts


def _generate_single_faq(client: OpenAI, text: str) -> Dict[str, str]:
    """Call GPT to generate a single Q&A pair from text."""
    system = "あなたはFAQ作成アシスタントです。JSON形式で1つの質問と回答を生成してください。"
    user = f"以下のテキストからよくある質問を1つ想定し、回答を作成してください。JSON形式で返してください。\n{text}"
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=300,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return {"question": data.get("question", ""), "answer": data.get("answer", "")}
    except Exception:
        return {"question": "", "answer": ""}


def generate_faqs_from_chunks(kb_name: str, *, client: Optional[OpenAI] = None) -> List[str]:
    """Generate FAQ files from existing chunks.

    Returns list of created file paths.
    """
    if client is None:
        key = ensure_openai_key()
        client = OpenAI(api_key=key)

    chunks = _load_chunks(kb_name)
    faq_dir = BASE_KNOWLEDGE_DIR / kb_name / "faq"
    faq_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []

    for cid, text in chunks.items():
        faq = _generate_single_faq(client, text)
        faq_data = {
            "question": faq.get("question", ""),
            "answer": faq.get("answer", ""),
            "generated_at": datetime.now().isoformat(),
            "source_chunk": cid,
        }
        path = faq_dir / f"faq_{uuid.uuid4().hex}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(faq_data, f, ensure_ascii=False, indent=2)
        created.append(str(path))

    return created


def load_faqs(kb_name: str) -> List[Dict[str, Any]]:
    """Load all FAQ JSON objects for a knowledge base."""
    faq_dir = BASE_KNOWLEDGE_DIR / kb_name / "faq"
    faqs = []
    if faq_dir.exists():
        for p in sorted(faq_dir.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    faqs.append(json.load(f))
            except Exception:
                continue
    return faqs
