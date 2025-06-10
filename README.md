# KnowledgePlus

This repository contains two related Streamlit applications:

- **knowledge_gpt_app** – a knowledge retrieval chatbot powered by GPT.
- **mm_kb_builder** – tools for building multimodal knowledge bases.

Each folder includes its own `requirements.txt`. A consolidated list of
all dependencies is provided at the repository root for convenience.

## Knowledge base location

Both apps now use a single `knowledge_base/` directory at the repository root.
If you were storing data in the previous `rag_knowledge_base` or
`multimodal_data` folders, move those contents into `knowledge_base/` before
upgrading.

## Installation

Create a virtual environment and install the packages:

```bash
pip install -r requirements.txt
```

Alternatively, install dependencies for each component separately:

```bash
# For the chatbot app
pip install -r knowledge_gpt_app/requirements.txt

# For the knowledge base builder
pip install -r mm_kb_builder/requirements.txt
```


## Configuration

Set the `OPENAI_API_KEY` environment variable so the applications can access the OpenAI API:

```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# macOS/Linux
export OPENAI_API_KEY=your_api_key_here
```

The applications will fail to start if this variable is missing.

## Running the app

Launch the unified interface using the helper scripts at the repository root. These scripts run `unified_app.py` which accepts documents, images and CAD files in one place:

```bash
# Windows
run_app.bat

# macOS/Linux
./run_app.sh
```


## Integration Plan

See [docs/integration_plan.md](docs/integration_plan.md) for an overview of the integration phases.

### Phase 3 Notes

Processed chunks, embeddings and metadata are now stored under
`knowledge_base/<kb_name>` using `save_processed_data()`. The metadata JSON
includes these paths so that other components can access the original files.
Whenever a chunk is stored the chatbot search index is refreshed automatically.

If you upload a file with the same name but different content, a version
number is appended so the existing file is preserved. To rebuild the search
index for an existing knowledge base you can run:

```bash
python reindex_kb.py <kb_name>
```

This reloads all chunks from disk and regenerates the BM25 index.
