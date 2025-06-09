#!/bin/bash
# Launch the unified Streamlit interface
cd "$(dirname "$0")" || exit 1
streamlit run knowledge_gpt_app/app.py
read -n 1 -s -r -p "Press any key to exit..."
echo

