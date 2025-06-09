# KnowledgePlus

KnowledgePlus consists of a GPT powered chatbot and a toolkit for building
multimodal knowledge bases.  Both applications require an OpenAI API key
provided via the `OPENAI_API_KEY` environment variable.

This repository contains two related Streamlit applications:

- **knowledge_gpt_app** – a knowledge retrieval chatbot powered by GPT.
- **mm_kb_builder** – tools for building multimodal knowledge bases.

Each folder includes its own `requirements.txt`. A consolidated list of all
dependencies is provided at the repository root for convenience.

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

## Usage

Set the `OPENAI_API_KEY` environment variable and then run the launcher
script for your platform:

```bash
run_app.bat  # Windows
# or
./run_app.sh # Linux/macOS
```

Both applications will read the API key from the environment when started.

## Knowledge Base Format

The apps share a unified directory layout for knowledge bases. Each
knowledge base resides in its own folder under a common root and includes
the following subdirectories:

```
[KB_ROOT]/<kb_name>/
├── chunks/       # extracted text chunks
├── metadata/     # per chunk metadata
├── embeddings/   # vector embeddings or indexes
├── files/        # source documents (optional)
├── images/       # images for multimodal items (optional)
└── kb_metadata.json
```

This structure allows both applications to read and update the same
knowledge sources.

