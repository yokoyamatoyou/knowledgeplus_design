# Integration Plan

This document outlines how the two applications in the repository work together.

1. **Create Knowledge Bases**
   - Use `mm_kb_builder` to upload images, PDFs, and CAD files and convert them into a unified knowledge base under `multimodal_data/`.
2. **Chat With the Data**
   - Run `knowledge_gpt_app` to query the generated knowledge base via GPT models.

Both components rely on the same directory structure so they can share embeddings and metadata seamlessly.
