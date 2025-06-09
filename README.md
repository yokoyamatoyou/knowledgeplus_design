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


## Integration Plan

See [docs/integration_plan.md](docs/integration_plan.md) for an overview of the integration phases.
