@echo off
REM Launch the unified Streamlit interface
cd /d "%~dp0"
streamlit run knowledge_gpt_app/app.py
pause

