import streamlit as st
import uuid
import base64
import types
import torch

# Ensure Streamlit page configuration is applied once across modules
if "_page_configured" not in st.session_state:
    st.set_page_config(
        page_title="Unified Knowledge Upload",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state["_page_configured"] = True

# Workaround: avoid Streamlit watcher errors with torch dynamic modules
if hasattr(torch, "classes") and not hasattr(torch.classes, "__path__"):
    torch.classes.__path__ = types.SimpleNamespace(_path=[])

from knowledge_gpt_app.app import (
    read_file,
    semantic_chunking,
    get_openai_client,
    apply_intel_theme,
)
from shared.search_utils import search_kb, refresh_search_engine
from mm_kb_builder.app import (
    process_cad_file,
    encode_image_to_base64,
    analyze_image_with_gpt4o,
    create_comprehensive_search_chunk,
    get_embedding,
    save_unified_knowledge_item,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_CAD_TYPES,
)
from generate_faq import generate_faqs_from_chunks

# Apply common theme styling
apply_intel_theme()

st.sidebar.title("Menu")
mode = st.sidebar.radio("Mode", ["Upload", "Search", "Chat", "FAQ"], index=0)

st.title("Unified Knowledge App")

kb_name = st.sidebar.text_input("Knowledge Base Name", "unified_kb")

if mode == "Upload":
    all_types = [
        'pdf', 'docx', 'xlsx', 'xls', 'txt'
    ] + SUPPORTED_IMAGE_TYPES + SUPPORTED_CAD_TYPES
    uploaded_files = st.file_uploader(
        "Upload Files",
        type=all_types,
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process Files"):
        client = get_openai_client()
        if not client:
            st.error("OpenAI client unavailable")
        else:
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    ext = file.name.split('.')[-1].lower()
                    bytes_data = file.getvalue()
                    file.seek(0)

                    if ext in ['pdf', 'docx', 'xlsx', 'xls', 'txt']:
                        text = read_file(file)
                        if text:
                            semantic_chunking(
                                text,
                                15,
                                'C',
                                'auto',
                                kb_name,
                                client,
                                original_filename=file.name,
                                original_bytes=bytes_data,
                            )
                            st.success(f"Processed text file {file.name}")
                        else:
                            st.error(f"Failed to read {file.name}")
                    elif ext in SUPPORTED_IMAGE_TYPES + SUPPORTED_CAD_TYPES:
                        if ext in SUPPORTED_CAD_TYPES:
                            img_b64, cad_meta = process_cad_file(file, ext)
                            if img_b64 is None:
                                st.error(f"CAD processing failed: {cad_meta.get('error')}")
                                continue
                        else:
                            img_b64 = encode_image_to_base64(file)
                            cad_meta = None
                        analysis = analyze_image_with_gpt4o(img_b64, file.name, cad_meta, client)
                        if "error" in analysis:
                            st.error(f"Analysis failed for {file.name}: {analysis['error']}")
                            continue
                        chunk = create_comprehensive_search_chunk(analysis, {})
                        embedding = get_embedding(chunk, client)
                        if embedding is None:
                            st.error(f"Embedding failed for {file.name}")
                            continue
                        success, _ = save_unified_knowledge_item(
                            str(uuid.uuid4()),
                            analysis,
                            {},
                            embedding,
                            file.name,
                            img_b64,
                            original_bytes=bytes_data,
                        )
                        if success:
                            st.success(f"Processed image/CAD file {file.name}")
                        else:
                            st.error(f"Saving failed for {file.name}")
                    else:
                        st.warning(f"Unsupported file type: {file.name}")

elif mode == "Search":
    query = st.text_input("Search Query")
    top_k = st.slider("Results", 1, 10, 5)
    threshold = st.slider("Threshold", 0.0, 1.0, 0.15, 0.01)
    if st.button("Search") and query:
        client = get_openai_client()
        if not client:
            st.error("OpenAI client unavailable")
        else:
            with st.spinner("Searching..."):
                results, not_found = search_kb(
                    kb_name,
                    query,
                    top_k=top_k,
                    threshold=threshold,
                    client=client,
                )
        if not results:
            st.info("No results found")
        else:
            for r in results:
                st.markdown(f"**{r['id']}** - {r['similarity']:.3f}")
                st.write(r['text'])

elif mode == "FAQ":
    st.subheader("Generate FAQs")
    max_tokens = st.number_input("Max GPT tokens", 100, 4000, 1000, 100)
    num_pairs = st.number_input("Q&A pairs", 1, 10, 3, 1)
    if st.button("FAQ生成"):
        client = get_openai_client()
        if not client:
            st.error("OpenAI client unavailable")
        else:
            with st.spinner("Generating FAQs..."):
                count = generate_faqs_from_chunks(kb_name, max_tokens, num_pairs, client=client)
                refresh_search_engine(kb_name)
            st.success(f"{count} FAQs created")

else:
    st.info("Chat mode is not implemented yet.")
