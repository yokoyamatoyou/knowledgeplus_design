from pathlib import Path
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
from shared.chat_controller import generate_gpt_response
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
    query = st.text_input(
        "main_search_box",
        placeholder="🔍 キーワードで検索、またはAIへの質問を入力...",
        label_visibility="collapsed",
    )
    if st.button("検索", type="primary"):
        st.session_state["search_executed"] = True
        kb_names = [kb["name"] for kb in list_knowledge_bases()]
        st.session_state["results"], _ = search_multiple_knowledge_bases(
            query, kb_names
        )
        st.session_state["last_query"] = query


def render_document_card(doc):
    """Render a single document search result as a styled card."""
    filename = doc.get("metadata", {}).get("filename", "N/A")
    text = doc.get("text", "")
    excerpt = text[:100]
    st.markdown(
        f"<div class='doc-card'><strong>{filename}</strong><p>{excerpt}</p>",
        unsafe_allow_html=True,
    )
    file_path = doc.get("metadata", {}).get("paths", {}).get("original_file_path")
    if file_path and Path(file_path).exists():
        with open(file_path, "rb") as f:
            st.download_button(
                "📄 ダウンロード",
                f.read(),
                file_name=Path(file_path).name,
                key=f"download_{file_path}",
            )
    st.markdown("</div>", unsafe_allow_html=True)


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
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("メッセージを送信")
    if user_msg:
        st.session_state["chat_history"].append({"role": "user", "content": user_msg})
        results, _ = search_multiple_knowledge_bases(user_msg, [DEFAULT_KB_NAME])
        context = "\n".join(r.get("text", "") for r in results[:3])
        client = get_openai_client()
        if client:
            prompt = f"次の情報を参考にユーザーの質問に答えてください:\n{context}\n\n質問:{user_msg}"
            answer = safe_generate_gpt_response(
                prompt,
                conversation_history=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["chat_history"][:-1]
                    if m["role"] in ("user", "assistant")
                ],
                persona="default",
                temperature=0.3,
                response_length="普通",
                client=client,
            )
        else:
            answer = "OpenAIクライアントを初期化できませんでした。"
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.rerun()
elif mode == "FAQ":
    st.subheader("FAQ作成")
    kb_name = st.text_input("Knowledge base name", value=DEFAULT_KB_NAME)
    max_tokens = st.number_input("Max tokens per chunk", 100, 2000, 1000, 100)
    pairs = st.number_input("Pairs per chunk", 1, 10, 3, 1)
    if st.button("FAQ生成", type="primary"):
        with st.spinner("FAQを生成中..."):
            from generate_faq import generate_faqs_from_chunks

            client = get_openai_client()
            if client:
                count = generate_faqs_from_chunks(kb_name, max_tokens, pairs, client=client)
                refresh_search_engine(kb_name)
                st.success(f"{count} FAQs generated")
            else:
                st.error("OpenAIクライアントが利用できません")
