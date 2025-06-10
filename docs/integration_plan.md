# Integration Plan

This document outlines the major phases for integrating the knowledge base builder with the Knowledge GPT application.

## Analysis
- **Objective:** Review existing code and define the overall approach for combining the two apps.
- **Key tasks:**
  - Inspect repository structure and dependencies.
  - Identify overlapping functionality and integration points.
  - Decide on shared configuration and environment setup.

## Upload and Processing
- **Objective:** Handle user uploads and convert them into a format suitable for search.
- **Key tasks:**
  - Accept text and image files via the interface.
  - Run OCR and text extraction if needed.
  - Segment and embed content for efficient retrieval.

## Knowledge Base Integration
- **Objective:** Store processed items in a unified knowledge base.
- **Key tasks:**
  - Save metadata and embeddings to the local storage layer.
  - Maintain links between original files and processed data.
  - Update indexing so that the chatbot can access new entries.

### Phase 3 Implementation

`save_processed_data()` now writes files under `knowledge_base/<kb_name>` and
returns their paths. These paths are stored inside each chunk's metadata JSON so
that the chatbot can provide download links. When a chunk or image is saved the
corresponding search index is refreshed via `refresh_search_engine()`.

If a file with the same name already exists, the new one will be saved with a
version suffix so older uploads remain intact. A standalone `reindex_kb.py`
script is provided to rebuild indexes from disk:

```bash
python reindex_kb.py <kb_name>
```

## FAQ Generation
- **Objective:** Automatically create frequently asked questions from the knowledge base.
- **Key tasks:**
  - Analyze content for common topics.
  - Generate question and answer pairs with GPT.
  - Store generated FAQs for reference within the app.

### Phase 4 Implementation

FAQ entries are generated from existing chunks using `generate_faq.py` or via the
sidebar button in `unified_app.py`. The script writes a `faqs.json` file under
`knowledge_base/<kb_name>` and stores embeddings so the search engine can index
them. In the UI a progress spinner is shown and the search index is refreshed
after generation.

Example CLI usage:

```bash
python generate_faq.py my_kb --max-tokens 500 --pairs 5
```

In the Streamlit interface, enter a knowledge base name and click **FAQ生成** to
produce FAQs with the configured token and pair counts.

## Local Storage
- **Objective:** Keep all data on disk so that the app can run offline.
- **Key tasks:**
  - Organize uploads, metadata and embeddings in a predictable directory layout.
  - Provide utilities for backup and restore.

## User Interface
- **Objective:** Offer a streamlined UI that unifies upload, search and chat features.
- **Key tasks:**
  - Merge builder and chatbot components into a single Streamlit app.
  - Expose upload and FAQ options alongside the chat interface.

## Testing
- **Objective:** Verify that each phase works as expected and that the combined app remains stable.
- **Key tasks:**
  - Unit test new utilities and data handling logic.
  - Run end-to-end tests for the full upload-to-chat workflow.
  - Collect feedback and fix any issues.

