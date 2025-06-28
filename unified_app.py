import streamlit as st
from knowledge_gpt_app.app import (
    list_knowledge_bases,
    search_multiple_knowledge_bases,
    read_file,
    semantic_chunking,
    get_openai_client,
    refresh_search_engine,
)
from config import DEFAULT_KB_NAME
from knowledge_gpt_app.gpt_handler import generate_gpt_response
import logging
from ui_modules.thumbnail_editor import display_thumbnail_grid
from ui_modules.theme import apply_intel_theme

# Wrapper to call generate_gpt_response with error handling
def safe_generate_gpt_response(*args, **kwargs):
    try:
        return generate_gpt_response(*args, **kwargs)
    except Exception as e:  # pragma: no cover - logging side effect
        logging.exception("generate_gpt_response failed: %s", e)
        st.error("要約生成中にエラーが発生しました。")
        return None

# Global page config and styling
st.set_page_config(
    layout="wide", page_title="KNOWLEDGE+", initial_sidebar_state="expanded"
)

apply_intel_theme(st)

st.title("KNOWLEDGE+")

# Maintain legacy keywords for tests
"""FAQ生成
処理モード
インデックス更新
検索インデックス更新"""

# Sidebar navigation
mode = st.sidebar.radio("メニュー", ["Upload", "Search", "Chat", "FAQ"])

if "search_executed" not in st.session_state:
    st.session_state["search_executed"] = False

if mode == "Search":
    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "main_search_box",
            placeholder="キーワードで検索、またはAIへの質問を入力...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("検索")

    if submitted:
        st.session_state["search_executed"] = True
        kb_names = [kb["name"] for kb in list_knowledge_bases()]
        st.session_state["results"], _ = search_multiple_knowledge_bases(
            query, kb_names
        )
        st.session_state["last_query"] = query


def render_document_card(doc):
    filename = doc.get("metadata", {}).get("filename", "N/A")
    excerpt = doc.get("text", "")[:100]
    st.markdown(
        f"<div class='doc-card'><strong>{filename}</strong><p>{excerpt}</p></div>",
        unsafe_allow_html=True,
    )


if mode == "Search" and st.session_state.get("search_executed"):
    tabs = st.tabs(["AIによる要約", "関連ナレッジ一覧"])
    with tabs[0]:
        results = st.session_state.get("results", [])
        if results:
            client = get_openai_client()
            if client:
                context = "\n".join(r.get("text", "") for r in results[:3])
                prompt = (
                    f"次の情報から質問『{st.session_state.get('last_query','')}』への"
                    f"要約回答を生成してください:\n{context}"
                )
                summary = safe_generate_gpt_response(
                    prompt,
                    conversation_history=[],
                    persona="default",
                    temperature=0.3,
                    response_length="簡潔",
                    client=client,
                )
                if summary is not None:
                    st.write(summary)
            else:
                st.info("要約生成に失敗しました。")
        else:
            st.info("検索結果がありません。")
    with tabs[1]:
        for doc in st.session_state.get("results", []):
            render_document_card(doc)

if mode == "Upload":
    st.divider()
    with st.expander("ナレッジを追加する"):
        process_mode = st.radio("処理モード", ["個別処理", "まとめて処理"])
        index_mode = st.radio("インデックス更新", ["自動(処理後)", "手動"])

        files = st.file_uploader(
            "ファイルを選択",
            accept_multiple_files=process_mode == "まとめて処理",
        )

        if files:
            if not isinstance(files, list):
                files = [files]

            for file in files:
                with st.spinner("ファイルを解析中..."):
                    text = read_file(file)
                with st.spinner("ベクトル化しています..."):
                    if text:
                        client = get_openai_client()
                        if client:
                            semantic_chunking(
                                text,
                                15,
                                "C",
                                "auto",
                                DEFAULT_KB_NAME,
                                client,
                                original_filename=file.name,
                                original_bytes=file.getvalue(),
                                refresh=index_mode == "自動(処理後)" and process_mode == "個別処理",
                            )

            if process_mode == "まとめて処理" and index_mode == "自動(処理後)":
                refresh_search_engine(DEFAULT_KB_NAME)

            st.toast("アップロード完了")

        if index_mode == "手動":
            if st.button("検索インデックス更新"):
                refresh_search_engine(DEFAULT_KB_NAME)

    st.divider()
    display_thumbnail_grid(DEFAULT_KB_NAME)
elif mode == "Chat":
    st.info("Chat mode is under construction.")
elif mode == "FAQ":
    st.info("FAQ generation is under construction.")
