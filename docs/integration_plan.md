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

### Phase 3 Upload Consolidation

The upload interface is unified with a single **Upload** button that accepts
multiple files at once. File types are detected by extension and processed
automatically:

- **PDF** – extract text with PyPDF2. If a page has no text, convert it to an
  image and run OCR so scanned PDFs are also handled.
- **DOCX** – read all paragraphs and run OCR on any embedded images so nothing
  is missed.
- **Excel** – iterate through every sheet. Append all cell text (formulas are
  replaced by their computed values) to the text representation before
  chunking. Images inside a sheet are extracted and processed with OCR so table
  screenshots are not lost.
- **Text with images** – when a document format mixes text and pictures (for
  example a Markdown file with screenshots) the text portion is extracted
  normally and each image is run through OCR. The results are appended to the
  same chunk sequence so searches cover both sources.

Images and CAD files retain the existing flow where optional metadata can be
entered manually after upload.

During a single upload session each file is processed sequentially under
`st.spinner` so progress is visible. Thumbnails will later be displayed for all
files so the user can review them before continuing.

## FAQ Generation
- **Objective:** Automatically create frequently asked questions from the knowledge base.
- **Key tasks:**
  - Analyze content for common topics.
  - Generate question and answer pairs with GPT.
  - Store generated FAQs for reference within the app.

### Phase 4 Implementation

FAQ entries are generated from existing chunks using `generate_faq.py` or via the
Streamlit interface. A dedicated **FAQ作成** mode has been added to
`knowledge_gpt_app` where you can select a knowledge base and generate FAQs.
The script writes a `faqs.json` file under `knowledge_base/<kb_name>` and stores
embeddings so the search engine can index them. In the UI a progress spinner is
shown and the search index is refreshed after generation.

Example CLI usage:

```bash
python generate_faq.py my_kb --max-tokens 500 --pairs 5
```

In the Streamlit interface, enter a knowledge base name and click **FAQ生成** to
produce FAQs with the configured token and pair counts.

### Phase 5 Implementation

The unified app now applies a shared Intel-inspired theme and displays a
`st.spinner` while uploads are processed. A `torch.classes` workaround prevents
reload issues during development. These adjustments polish the user interface
without altering existing data or indexes.

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

### Phase 1 Design Review

- **Objective:**
  - Map the current features of `unified_app.py` and `knowledge_gpt_app/app.py` to
    decide how they should be merged.
  - Define a unified layout where the sidebar is always visible.

- **Tasks:**
  - **Code investigation**
    - `unified_app.py` handles uploads by calling helpers such as
      `read_file`, `semantic_chunking`, `process_cad_file` and
      `save_unified_knowledge_item`.
      It does not yet include a search screen.
    - In `knowledge_gpt_app` the modules `knowledge_search.py` and `conversation.py`
      implement search and chat. These overlap with the upload script and need to
      be unified.
  - **UI policy**
    - Keep a left sidebar visible in every mode including the upload screen.
    - Provide menu items: **Upload**, **Search**, **Chat**, **FAQ** so users can
      jump between modes.
  - **Navigation sketch**

    ```
    [Sidebar]
      ├─ Upload -> File upload
      ├─ Search -> Knowledge search
      ├─ Chat   -> Conversation interface
      └─ FAQ    -> FAQ generation
    ```

  - No code will be changed during this phase. Results will be used as a basis
    for later implementation steps.

### Phase 4 Thumbnail & Metadata UI

- **Objective:**
  - Show uploaded items as thumbnails and enable per-item metadata entry.
- **Thumbnail spec:**
  - Items are arranged in a 3x3 grid (max 9 per page).
  - Document thumbnails use 10pt text with 12 characters taken from the middle
    of a chunk.
  - If more than 9 items exist, navigation buttons allow page changes.
- **Metadata flow:**
  - Clicking a thumbnail reveals fields for `title` and `tags`.
  - Saving writes a JSON file to `knowledge_base/<kb_name>/metadata/<id>_user.json`.
  - The `id` corresponds to the internal UUID assigned at upload time.
- **UI mock:**

  ```
  [prev] Page 1/2 [next]
  ┌───┬───┬───┐
  │ 1 │ 2 │ 3 │
  ├───┼───┼───┤
  │ 4 │ 5 │ 6 │
  ├───┼───┼───┤
  │ 7 │ 8 │ 9 │
  └───┴───┴───┘
  ```
  Selecting a cell opens a small form below the grid:

  ```
  Title: [__________]
  Tags : [__________]
  [ Save ]
  ```

