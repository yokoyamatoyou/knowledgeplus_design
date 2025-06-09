# KnowledgePlus

This repository contains two related Streamlit applications:

- **knowledge_gpt_app** – a knowledge retrieval chatbot powered by GPT.
- **mm_kb_builder** – tools for building multimodal knowledge bases.

Each folder includes its own `requirements.txt`. A consolidated list of
all dependencies is provided at the repository root for convenience.

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

## Purpose

`KnowledgePlus` is for building and searching multimodal knowledge bases.
The `mm_kb_builder` app ingests images, PDF documents and CAD files to
create searchable embeddings.

## Running the Multimodal KB Builder

```bash
cd mm_kb_builder
streamlit run app.py
```

Set the `OPENAI_API_KEY` environment variable before launching so the
app can call OpenAI APIs.

### Expected directory layout

When you run the builder the following structure is created under
`multimodal_data/`:

```
multimodal_data/
└── <kb_name>/
    ├── files/
    ├── images/
    ├── chunks/
    ├── embeddings/
    ├── metadata/
    └── kb_metadata.json
```

This directory is shared with the chatbot component.

See the [integration plan](integration_plan.md) for how the two apps
fit together.

