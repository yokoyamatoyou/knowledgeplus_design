# KnowledgePlus Agent Blueprint

This document outlines the overall plan and working rules for AI agents
contributing to this repository. Practical setup instructions such as how to
install dependencies, run the app and execute the test suite live in
`README.md`. The blueprint below expands on the roles, phase goals and UI
expectations so that each contributor understands exactly how the system should
behave.

## 1. Project Overview

### 1.1 Name
KnowledgePlus

### 1.2 Goal
Provide a unified platform where documents, images and CAD files can be indexed
and queried through GPT.  Users interact via a streamlined interface with
sliders, buttons and preview thumbnails.  FAQs generated from uploaded content
are stored for future reference.

### 1.3 Development Philosophy
- Follow the AIDE (Agent‑Integrated Development Environment) approach so agents
  can autonomously code new features.
- Keep the application in a working state at all times.  After each phase the
  code should run without manual fixes.
- Update `requirements.txt` whenever new libraries are introduced.
- The OpenAI API key must be set via `OPENAI_API_KEY` before running the app.
- `run_app.sh` launches the main Streamlit interface and should be used for
  local testing.

## 2. Agent Team

### 2.1 Knowledge Architect
- **Goal:** build a structured knowledge base from any input format.
- **Tools:** LangChain, Unstructured, PyPDF2, Pillow, OpenCV, trimesh, FAISS and
  others as needed.
- **Primary tasks:**
  - implement file loaders in `unified_app.py` to accept `.txt`, `.pdf`, `.pptx`,
    `.jpg`, `.png`, `.step`, `.iges` and related types;
  - split documents into semantic chunks in `vector_store.py` using user defined
    size and overlap settings;
  - embed chunks with OpenAI embeddings and save them in the FAISS store along
    with metadata;
  - generate image captions via `mm_kb_builder/app.py` so screenshots can be
    searched;
  - provide `reindex_kb.py` to rebuild indexes when existing data changes.

### 2.2 Retrieval Specialist
- **Goal:** return relevant chunks quickly and accurately for a given query.
- **Tools:** LangChain, FAISS, NumPy, OpenAI embeddings.
- **Primary tasks:**
  - vectorise user queries in `knowledge_search.py`;
  - perform FAISS searches with a similarity threshold set by the UI slider;
  - re-rank and filter results in `unified_app.py` to remove noise;
  - extract source information (file name, page, snippet) so it can be shown in
    the preview pane.

### 2.3 Content Synthesis Writer
- **Goal:** combine search results into helpful summaries, FAQs and chat replies.
- **Tools:** OpenAI GPT‑4.1 and mini variants via LangChain.
- **Primary tasks:**
  - generate concise answers in `gpt_handler.py` based on search results;
  - analyse entire knowledge bases in `generate_faq.py` to create FAQ
    question/answer pairs and store them as `faqs.json`;
  - maintain conversation history in `conversation.py` so chat context is
    preserved across turns;
  - adjust the tone of replies according to the user selected persona.

### 2.4 Interface Designer
- **Goal:** deliver an intuitive Streamlit UI covering upload, search, FAQ and
  chat modes.
- **Tools:** Streamlit components plus simple HTML/CSS and Plotly for
  visualisation.
- **Primary tasks:**
  - organise the main screen so users can switch between **Upload**, **Search**,
    **Chat** and **FAQ** without losing their place;
  - place `st.expander`, `st.slider`, `st.button` and `st.spinner` components
    with clear labels;
  - resolve `Expanders may not be nested` errors by keeping layouts flat;
  - show progress bars during lengthy operations and display thumbnails for
    uploaded images and CAD files.

## 3. Development Roadmap

1. **Phase 1 – Core engine stabilisation & MVP**
   - Refactor `unified_app.py` into modular components such as
     `knowledge_builder.py`, `retriever.py`, `content_generator.py` and
     `ui_manager.py`.
   - Fix critical errors recorded in `rag_tool.log` so the app runs cleanly.
   - Add unit tests in `tests/` for each major module.
   
2. **Phase 2 – Multimodal support & usability**
   - Integrate the multimodal builder to handle images and CAD files alongside
     text.
   - Show thumbnails in the search results and allow previewing original files.
   - Persist user settings such as chunk size, overlap and similarity threshold
     via `st.session_state`.
   
3. **Phase 3 – Advanced features & optimisation**
   - Improve FAQ generation for higher quality and faster results.
   - Introduce hybrid search and caching to boost performance.
   - Provide tools to export and resume chat history.
   
4. **Phase 4 – Autonomy & deployment**
   - Auto‑install missing dependencies when launching with `run_app.sh`.
   - Optimise `build_exe.py` so the app can be distributed as a single binary.
   - Expand documentation and conduct usability tests with new users.

## 4. Coding, Testing and PR Guidelines

- Follow PEP 8 and keep functions small with clear names.
- Provide both batch and individual upload modes with optional FAQ generation
  and manual index refresh.
- Run tests before committing:
  ```bash
  pytest -q
  ```
  Sample data is under `tests/data`.
- Commit messages should be concise and in English. Include a summary of what
  changed and avoid generic wording such as "apply patch".
- Every pull request must include a **Summary** section describing the main
  updates and a **Testing** section showing the `pytest` results.
- Keep the working tree clean by committing all modified files before finishing.

