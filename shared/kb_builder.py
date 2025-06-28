import uuid
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

from .file_processor import FileProcessor
from .upload_utils import save_processed_data
from config import DEFAULT_KB_NAME

logger = logging.getLogger(__name__)


class KnowledgeBuilder:
    """Build knowledge items from uploaded files."""

    def __init__(self, file_processor: FileProcessor | None = None) -> None:
        """Set up helpers from mm_kb_builder.app lazily to avoid circular imports."""
        self.file_processor = file_processor or FileProcessor()
        from mm_kb_builder import app as kb_app

        self.encode_image_to_base64 = kb_app.encode_image_to_base64
        self.analyze_image_with_gpt4o = kb_app.analyze_image_with_gpt4o
        self.process_cad_file = kb_app.process_cad_file
        self.create_comprehensive_search_chunk = kb_app.create_comprehensive_search_chunk
        self.get_embedding = kb_app.get_embedding
        self.save_unified_knowledge_item = kb_app.save_unified_knowledge_item
        self.SUPPORTED_IMAGE_TYPES = kb_app.SUPPORTED_IMAGE_TYPES
        self.SUPPORTED_CAD_TYPES = kb_app.SUPPORTED_CAD_TYPES
        self.get_openai_client = kb_app.get_openai_client
        self.refresh_search_engine = kb_app._refresh_search_engine

    def build_from_file(
        self,
        uploaded_file,
        *,
        analysis: Dict[str, Any] | None = None,
        image_base64: str | None = None,
        user_additions: Dict[str, Any] | None = None,
        cad_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:
        """Process and store a single uploaded file and return saved metadata."""

        file_extension = uploaded_file.name.split(".")[-1].lower()
        client = self.get_openai_client()
        if client is None:
            logger.error("OpenAI client unavailable")
            return None

        if file_extension in self.SUPPORTED_IMAGE_TYPES + self.SUPPORTED_CAD_TYPES:
            is_cad = file_extension in self.SUPPORTED_CAD_TYPES
            if analysis is None:
                if is_cad:
                    image_base64, cad_metadata = self.process_cad_file(uploaded_file, file_extension)
                else:
                    image_base64 = image_base64 or self.encode_image_to_base64(uploaded_file)
            if not image_base64:
                return None
            if analysis is None:
                analysis = self.analyze_image_with_gpt4o(image_base64, uploaded_file.name, cad_metadata, client)
                if "error" in analysis:
                    return None
            user_additions = user_additions or {}
            chunk_id = str(uuid.uuid4())
            search_chunk = self.create_comprehensive_search_chunk(analysis, user_additions)
            embedding = self.get_embedding(search_chunk, client)
            if embedding is None:
                return None
            success, saved_item = self.save_unified_knowledge_item(
                chunk_id,
                analysis,
                user_additions,
                embedding,
                uploaded_file.name,
                image_base64,
                original_bytes=uploaded_file.getvalue(),
            )
            return saved_item if success else None

        # Text based documents
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)
        uploaded_file.seek(0)
        text = self.file_processor.process_file(tmp_path)
        tmp_path.unlink(missing_ok=True)
        if not text:
            return None
        embedding = self.get_embedding(text, client)
        if embedding is None:
            return None
        chunk_id = str(uuid.uuid4())
        metadata = {"filename": uploaded_file.name, "type": "text_chunk"}
        paths = save_processed_data(
            DEFAULT_KB_NAME,
            chunk_id,
            chunk_text=text,
            embedding=embedding,
            metadata=metadata,
            original_filename=uploaded_file.name,
            original_bytes=uploaded_file.getvalue(),
        )
        self.refresh_search_engine(DEFAULT_KB_NAME)
        return {"id": chunk_id, **paths}


