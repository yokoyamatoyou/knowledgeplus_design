# KnowledgePlus

This repository combines a knowledge retrieval chatbot and a multimodal knowledge base builder.

## Installation

Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

Set your OpenAI API key so the apps can access the API:

```bash
# macOS/Linux
export OPENAI_API_KEY=your_api_key_here

# Windows
set OPENAI_API_KEY=your_api_key_here
```

## Running the Application

Use the helper scripts at the repository root to launch the unified interface:

```bash
# macOS/Linux
./run_app.sh

# Windows
run_app.bat
```

These scripts run `unified_app.py`, which provides a single entry point for both the chatbot and knowledge base builder.

## Repository Components

- **knowledge_gpt_app** – Streamlit chatbot for searching the knowledge base and chatting with GPT.
- **mm_kb_builder** – Tools for building a multimodal knowledge base from text, images and CAD files.
- **unified_app.py** – Central interface that ties both components together.

Design details for the interface and integration plan can be found in [ui_design_plan.md](ui_design_plan.md) and [docs/integration_plan.md](docs/integration_plan.md).

## Running Tests

Execute the test suite with:

```bash
pytest -q
```
