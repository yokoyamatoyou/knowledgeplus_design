import os
import json
import pickle
from pathlib import Path

# Base directory for all knowledge bases
BASE_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
BASE_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_openai_key():
    """Return the OpenAI API key or raise an informative error."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
    return api_key


def _ensure_dirs(kb_dir: Path):
    """Create standard subdirectories for a knowledge base."""
    subdirs = {}
    for name in ["chunks", "embeddings", "metadata", "images", "files"]:
        path = kb_dir / name
        path.mkdir(parents=True, exist_ok=True)
        subdirs[name] = path
    return subdirs


def save_processed_data(
    kb_name: str,
    chunk_id: str,
    chunk_text: str = None,
    embedding=None,
    metadata: dict = None,
    original_filename: str = None,
    original_bytes: bytes = None,
    image_bytes: bytes = None,
):
    """Save processed chunk, embedding and metadata in a unified layout."""
    kb_dir = BASE_KNOWLEDGE_DIR / kb_name
    dirs = _ensure_dirs(kb_dir)

    paths = {}

    if chunk_text is not None:
        chunk_path = dirs["chunks"] / f"{chunk_id}.txt"
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk_text)
        paths["chunk_path"] = str(chunk_path)

    if metadata is not None:
        meta = metadata.copy()
        if original_filename:
            meta.setdefault("original_file", original_filename)
        meta_path = dirs["metadata"] / f"{chunk_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        paths["metadata_path"] = str(meta_path)

    if embedding is not None:
        emb_path = dirs["embeddings"] / f"{chunk_id}.pkl"
        with open(emb_path, "wb") as f:
            pickle.dump({"embedding": embedding}, f)
        paths["embedding_path"] = str(emb_path)

    if image_bytes is not None:
        img_path = dirs["images"] / f"{chunk_id}.jpg"
        with open(img_path, "wb") as f:
            f.write(image_bytes)
        paths["image_path"] = str(img_path)

    if original_bytes is not None and original_filename:
        file_path = dirs["files"] / original_filename
        with open(file_path, "wb") as f:
            f.write(original_bytes)
        paths["original_file_path"] = str(file_path)

    return paths
