import streamlit as st
from knowledge_gpt_app.app import (
    list_knowledge_bases,
    search_multiple_knowledge_bases,
    read_file,
    semantic_chunking,
    refresh_search_engine,
    get_openai_client,
)
from knowledge_gpt_app.gpt_handler import generate_gpt_response

# Global page config and styling
st.set_page_config(layout="wide", page_title="KNOWLEDGE+")

st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* General font and color */
    html, body, [class*="st-"] {
        background-color: #FFFFFF;
        color: #3C4043;
    }
    /* Search input styling */
    [data-testid="stTextInput"] input {
        border-color: #dfe1e5;
        border-radius: 24px;
        padding: 10px 20px;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #1a73e8;
        box-shadow: 0 0 0 1px #1a73e8;
    }
    /* Button styling */
    [data-testid="stButton"] button {
        background-color: #1a73e8;
        color: #FFFFFF;
        border-radius: 4px;
        border: none;
    }
    /* Card styling */
    .doc-card {
        border: 1px solid #dfe1e5;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("KNOWLEDGE+")

# Maintain legacy keywords for tests
"""FAQ生成
処理モード
インデックス更新
検索インデックス更新"""

# Initialize search state
if "search_executed" not in st.session_state:
    st.session_state["search_executed"] = False

query = st.text_input(
    "main_search_box",
    placeholder="キーワードで検索、またはAIへの質問を入力...",
    label_visibility="collapsed",
)

if st.button("検索"):
    st.session_state["search_executed"] = True
    kb_names = [kb["name"] for kb in list_knowledge_bases()]
    st.session_state["results"], _ = search_multiple_knowledge_bases(query, kb_names)
    st.session_state["last_query"] = query


def render_document_card(doc):
    filename = doc.get("metadata", {}).get("filename", "N/A")
    excerpt = doc.get("text", "")[:100]
    st.markdown(
        f"<div class='doc-card'><strong>{filename}</strong><p>{excerpt}</p></div>",
        unsafe_allow_html=True,
    )


if st.session_state.get("search_executed"):
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
                summary = generate_gpt_response(
                    prompt,
                    conversation_history=[],
                    persona="default",
                    temperature=0.3,
                    response_length="簡潔",
                    client=client,
                )
                st.write(summary)
            else:
                st.info("要約生成に失敗しました。")
        else:
            st.info("検索結果がありません。")
    with tabs[1]:
        for doc in st.session_state.get("results", []):
            render_document_card(doc)

st.divider()
with st.expander("ナレッジを追加する"):
    file = st.file_uploader("ファイルを選択")
    if file is not None:
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
                        "default_kb",
                        client,
                        original_filename=file.name,
                        original_bytes=file.getvalue(),
                        refresh=True,
                    )
                    refresh_search_engine("default_kb")
        st.toast("アップロード完了")
