import streamlit as st
import uuid
import base64

st.set_page_config(page_title="Unified Knowledge Upload", layout="wide")

from knowledge_gpt_app.app import (
    read_file,
    semantic_chunking,
    get_openai_client,
    refresh_search_engine,
)
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

st.title("Unified Knowledge Upload")

kb_name = st.text_input("Knowledge Base Name", "unified_kb")

st.sidebar.header("Actions")
max_tokens = st.sidebar.number_input("Max GPT tokens", 100, 4000, 1000, 100)
num_pairs = st.sidebar.number_input("Q&A pairs", 1, 10, 3, 1)
if st.sidebar.button("FAQ生成"):
    client = get_openai_client()
    if not client:
        st.sidebar.error("OpenAI client unavailable")
    else:
        with st.sidebar.spinner("Generating FAQs..."):
            count = generate_faqs_from_chunks(kb_name, max_tokens, num_pairs, client=client)
            refresh_search_engine(kb_name)
        st.sidebar.success(f"{count} FAQs created")

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
