import os
import sys
import streamlit as st
from openai import OpenAI
import json
import base64
from PIL import Image
import io
import numpy as np
import pandas as pd
from datetime import datetime
import uuid
from pathlib import Path
import logging
import tempfile
import shutil
from shared.upload_utils import (
    save_processed_data,
    BASE_KNOWLEDGE_DIR as SHARED_KB_DIR,
    ensure_openai_key,
)
from knowledge_gpt_app.app import refresh_search_engine

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# インテル風デザインテーマ適用
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def apply_intel_theme():
    """インテル社風の洗練されたデザインテーマを適用"""
    st.markdown("""
    <style>
    /* インテルカラーパレット */
    :root {
        --intel-blue: #0071c5;
        --intel-dark-blue: #003c71;
        --intel-light-blue: #00c7fd;
        --intel-white: #ffffff;
        --intel-light-gray: #f5f5f5;
        --intel-gray: #e5e5e5;
        --intel-dark-gray: #666666;
    }
    
    /* フォント設定 */
    .main, .sidebar, [data-testid="stApp"] {
        font-family: 'Meiryo', 'Hiragino Kaku Gothic ProN', sans-serif;
    }
    
    /* メインヘッダー */
    .main h1 {
        color: var(--intel-dark-blue);
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, var(--intel-blue), var(--intel-light-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* セクションヘッダー */
    .main h2 {
        color: var(--intel-dark-blue);
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--intel-blue);
        padding-left: 1rem;
    }
    
    .main h3 {
        color: var(--intel-dark-blue);
        font-weight: 500;
        font-size: 1.3rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* サイドバー */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, var(--intel-white), var(--intel-light-gray));
        border-right: 2px solid var(--intel-gray);
    }
    
    .sidebar h2, .sidebar h3 {
        color: var(--intel-dark-blue);
        font-weight: 600;
    }
    
    /* タブシステム */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--intel-light-gray);
        border-radius: 10px;
        padding: 5px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background: transparent;
        border-radius: 6px;
        color: var(--intel-dark-gray);
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--intel-blue), var(--intel-dark-blue));
        color: var(--intel-white);
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(0, 113, 197, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--intel-light-blue);
        color: var(--intel-white);
        transform: translateY(-1px);
    }
    
    /* ボタンスタイリング */
    .stButton > button {
        background: linear-gradient(135deg, var(--intel-blue), var(--intel-dark-blue));
        color: var(--intel-white);
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 113, 197, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--intel-dark-blue), var(--intel-blue));
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 113, 197, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(0, 113, 197, 0.3);
    }
    
    /* セカンダリボタン */
    .stButton > button[kind="secondary"] {
        background: var(--intel-white);
        color: var(--intel-blue);
        border: 2px solid var(--intel-blue);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--intel-light-blue);
        color: var(--intel-white);
        border-color: var(--intel-light-blue);
    }
    
    /* 入力フィールド */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border: 2px solid var(--intel-gray);
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: var(--intel-white);
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--intel-blue);
        box-shadow: 0 0 0 3px rgba(0, 113, 197, 0.1);
        outline: none;
    }
    
    /* メトリクス */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--intel-white), var(--intel-light-gray));
        border: 1px solid var(--intel-gray);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--intel-dark-blue);
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: var(--intel-dark-gray);
        font-weight: 500;
    }
    
    /* エキスパンダー */
    .streamlit-expanderHeader {
        background: var(--intel-light-gray);
        border: 1px solid var(--intel-gray);
        border-radius: 6px;
        padding: 0.75rem;
        font-weight: 600;
        color: var(--intel-dark-blue);
    }
    
    .streamlit-expanderContent {
        background: var(--intel-white);
        border: 1px solid var(--intel-gray);
        border-top: none;
        border-radius: 0 0 6px 6px;
        padding: 1rem;
    }
    
    /* アラート */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stAlert[data-baseweb="notification"] {
        background: linear-gradient(135deg, var(--intel-light-blue), var(--intel-blue));
        color: var(--intel-white);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #00c851, #007e33);
        color: var(--intel-white);
    }
    
    .stError {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: var(--intel-white);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffbb33, #ff8800);
        color: var(--intel-white);
    }
    
    /* プログレスバー */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--intel-blue), var(--intel-light-blue));
        border-radius: 10px;
    }
    
    /* データフレーム */
    .dataframe {
        border: 1px solid var(--intel-gray);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, var(--intel-blue), var(--intel-dark-blue));
        color: var(--intel-white);
        font-weight: 600;
        padding: 1rem;
    }
    
    .dataframe td {
        padding: 0.75rem;
        border-bottom: 1px solid var(--intel-light-gray);
    }
    
    .dataframe tr:hover {
        background: var(--intel-light-gray);
    }
    
    /* ファイルアップロード */
    .stFileUploader > div > div {
        border: 2px dashed var(--intel-blue);
        border-radius: 10px;
        padding: 2rem;
        background: var(--intel-light-gray);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--intel-dark-blue);
        background: var(--intel-white);
    }
    
    /* カスタムコンテナ */
    .intel-container {
        background: var(--intel-white);
        border: 1px solid var(--intel-gray);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* フッター */
    .footer-text {
        text-align: center;
        color: var(--intel-dark-gray);
        font-size: 0.9rem;
        padding: 2rem 0;
        border-top: 1px solid var(--intel-gray);
        margin-top: 3rem;
        background: var(--intel-light-gray);
    }
    
    /* スピナー */
    .stSpinner > div {
        border-color: var(--intel-blue);
    }
    
    /* セレクトボックス */
    .stSelectbox > div > div {
        background: var(--intel-white);
        border: 2px solid var(--intel-gray);
        border-radius: 6px;
    }
    
    /* チェックボックス */
    .stCheckbox > label {
        color: var(--intel-dark-blue);
        font-weight: 500;
    }
    
    /* テキスト強調 */
    .main strong, .main b {
        color: var(--intel-dark-blue);
        font-weight: 700;
    }
    
    /* コードブロック */
    .stCodeBlock {
        background: var(--intel-light-gray);
        border: 1px solid var(--intel-gray);
        border-radius: 6px;
    }
    
    /* 画像 */
    .stImage {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* カラムセパレータ */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* セクション区切り */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--intel-blue), transparent);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ページ設定
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
if not st.session_state.get("_page_configured", False):
    st.set_page_config(
        page_title="マルチモーダルナレッジ構築ツール",
        layout="wide",
    )
    st.session_state["_page_configured"] = True

# インテル風テーマ適用
apply_intel_theme()

# ライブラリチェック
try:
    import pdf2image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("pdf2image が見つかりません。PDF処理機能は無効です。")

try:
    import pytesseract
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False

# CAD処理ライブラリ
try:
    import ezdxf
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Polygon
    DXF_SUPPORT = True
except ImportError:
    DXF_SUPPORT = False
    st.info("ezdxf/matplotlib が見つかりません。DXF処理機能は無効です。")

try:
    import trimesh
    STL_SUPPORT = True
except ImportError:
    STL_SUPPORT = False
    st.info("trimesh が見つかりません。STL処理機能は無効です。")

try:
    import cadquery as cq
    STEP_SUPPORT = True
except ImportError:
    STEP_SUPPORT = False
    st.info("cadquery が見つかりません。STEP処理機能は無効です。")

# ハイブリッド検索用
try:
    import faiss
    from rank_bm25 import BM25Okapi
    # Sudachi使用
    from sudachipy import tokenizer
    from sudachipy import dictionary
    ADVANCED_SEARCH = True
    TOKENIZER_TYPE = "sudachi"
except ImportError:
    try:
        # MeCab fallback
        import MeCab
        ADVANCED_SEARCH = True
        TOKENIZER_TYPE = "mecab"
    except ImportError:
        ADVANCED_SEARCH = False
        TOKENIZER_TYPE = None
        st.info("高度検索機能が無効です（推奨: sudachipy, または faiss + rank-bm25 + mecab-python3）")

# 設定
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 親ディレクトリ(リポジトリルート)をパスに追加
repo_root = os.path.dirname(current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('multimodal_kb_builder')

# 定数
GPT4O_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536  # コスト効率化（デフォルト3072→1536）
SUPPORTED_IMAGE_TYPES = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
SUPPORTED_DOCUMENT_TYPES = ['pdf']
SUPPORTED_CAD_TYPES = ['dxf', 'stl', 'ply', 'obj', 'step', 'stp', 'iges', 'igs', '3ds']

# 共通ナレッジベースディレクトリ
BASE_KNOWLEDGE_DIR = SHARED_KB_DIR
BASE_KNOWLEDGE_DIR.mkdir(exist_ok=True)

# データディレクトリ
DATA_DIR = BASE_KNOWLEDGE_DIR

# セッション状態初期化
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}
if 'current_editing_id' not in st.session_state:
    st.session_state.current_editing_id = None

# OpenAIクライアント取得
@st.cache_resource
def get_openai_client():
    try:
        api_key = ensure_openai_key()
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"OpenAIクライアント初期化エラー: {e}")
        return None

def encode_image_to_base64(image_file):
    """画像ファイルをbase64エンコード"""
    try:
        if hasattr(image_file, 'type') and image_file.type == "application/pdf":
            if not PDF_SUPPORT:
                st.error("PDF処理にはpdf2imageライブラリが必要です")
                return None
            images = pdf2image.convert_from_bytes(image_file.read())
            if images:
                buffered = io.BytesIO()
                images[0].save(buffered, format="PNG")
                image_file.seek(0)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            image_file.seek(0)
            image_bytes = image_file.read()
            image_file.seek(0)
            return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"画像base64エンコードエラー: {e}")
        st.error(f"画像の処理中にエラーが発生しました: {e}")
        return None

def process_dxf_file(dxf_file):
    """DXFファイルを処理して画像とメタデータを生成"""
    if not DXF_SUPPORT:
        return None, {"error": "DXF処理ライブラリが利用できません"}
    
    try:
        dxf_file.seek(0)
        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as temp_file:
            temp_file.write(dxf_file.read())
            temp_file_path = temp_file.name
        dxf_file.seek(0)
        
        doc = ezdxf.readfile(temp_file_path)
        msp = doc.modelspace()
        
        entities_info = {
            'lines': [],
            'circles': [],
            'arcs': [],
            'texts': [],
            'dimensions': [],
            'blocks': []
        }
        
        for entity in msp:
            if entity.dxftype() == 'LINE':
                entities_info['lines'].append({
                    'start': tuple(entity.dxf.start[:2]),
                    'end': tuple(entity.dxf.end[:2])
                })
            elif entity.dxftype() == 'CIRCLE':
                entities_info['circles'].append({
                    'center': tuple(entity.dxf.center[:2]),
                    'radius': entity.dxf.radius
                })
            elif entity.dxftype() == 'TEXT':
                entities_info['texts'].append({
                    'text': entity.dxf.text,
                    'position': tuple(entity.dxf.insert[:2])
                })
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for line in entities_info['lines']:
            ax.plot([line['start'][0], line['end'][0]], 
                   [line['start'][1], line['end'][1]], 'b-', linewidth=1)
        
        for circle in entities_info['circles']:
            circle_patch = Circle(circle['center'], circle['radius'], 
                                fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(circle_patch)
        
        for text in entities_info['texts']:
            ax.text(text['position'][0], text['position'][1], text['text'], 
                   fontsize=8, ha='left')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"DXF Drawing: {dxf_file.name}")
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        os.unlink(temp_file_path)
        
        metadata = {
            'file_type': 'DXF',
            'total_entities': len(list(msp)),
            'lines_count': len(entities_info['lines']),
            'circles_count': len(entities_info['circles']),
            'texts_count': len(entities_info['texts']),
            'layers': [layer.dxf.name for layer in doc.layers],
            'drawing_units': doc.header.get('$INSUNITS', 'Unknown'),
            'entities_detail': entities_info
        }
        
        return image_base64, metadata
        
    except Exception as e:
        logger.error(f"DXF処理エラー: {e}")
        return None, {"error": f"DXF処理中にエラーが発生しました: {e}"}

def process_stl_file(stl_file):
    """STLファイルを処理して複数角度の画像を生成"""
    if not STL_SUPPORT:
        return None, {"error": "STL処理ライブラリが利用できません"}
    
    try:
        stl_file.seek(0)
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as temp_file:
            temp_file.write(stl_file.read())
            temp_file_path = temp_file.name
        stl_file.seek(0)
        
        mesh = trimesh.load_mesh(temp_file_path)
        
        metadata = {
            'file_type': 'STL',
            'vertices_count': len(mesh.vertices),
            'faces_count': len(mesh.faces),
            'volume': float(mesh.volume),
            'surface_area': float(mesh.area),
            'bounds': mesh.bounds.tolist(),
            'center_mass': mesh.center_mass.tolist(),
            'is_watertight': mesh.is_watertight,
            'is_valid': mesh.is_valid
        }
        
        angles = [(0, 0), (90, 0), (0, 90), (45, 45)]
        images = []
        
        for i, (azimuth, elevation) in enumerate(angles):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot_trisurf(mesh.vertices[:, 0], 
                           mesh.vertices[:, 1], 
                           mesh.vertices[:, 2],
                           triangles=mesh.faces, 
                           alpha=0.8, cmap='viridis')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"STL Model - View {i+1} ({azimuth}°, {elevation}°)")
            ax.view_init(elev=elevation, azim=azimuth)
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='PNG', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            images.append(image_base64)
            plt.close()
        
        os.unlink(temp_file_path)
        
        return images[0], {**metadata, 'additional_views': images[1:]}
        
    except Exception as e:
        logger.error(f"STL処理エラー: {e}")
        return None, {"error": f"STL処理中にエラーが発生しました: {e}"}

def process_step_file(step_file):
    """STEPファイルを処理（CadQuery使用）"""
    if not STEP_SUPPORT:
        return None, {"error": "STEP処理ライブラリ（CadQuery）が利用できません"}
    
    try:
        step_file.seek(0)
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as temp_file:
            temp_file.write(step_file.read())
            temp_file_path = temp_file.name
        step_file.seek(0)
        
        result = cq.importers.importStep(temp_file_path)
        
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as stl_temp:
            stl_temp_path = stl_temp.name
        
        cq.exporters.export(result, stl_temp_path)
        
        with open(stl_temp_path, 'rb') as stl_temp_file:
            image_base64, metadata = process_stl_file(stl_temp_file)
        
        if metadata and 'error' not in metadata:
            metadata['file_type'] = 'STEP'
            metadata['original_format'] = 'STEP/STP'
        
        os.unlink(temp_file_path)
        os.unlink(stl_temp_path)
        
        return image_base64, metadata
        
    except Exception as e:
        logger.error(f"STEP処理エラー: {e}")
        return None, {"error": f"STEP処理中にエラーが発生しました: {e}"}

def process_cad_file(cad_file, file_extension):
    """CADファイルを形式に応じて処理"""
    file_ext = file_extension.lower()
    
    if file_ext == 'dxf':
        return process_dxf_file(cad_file)
    elif file_ext == 'stl':
        return process_stl_file(cad_file)
    elif file_ext in ['step', 'stp']:
        return process_step_file(cad_file)
    elif file_ext in ['ply', 'obj'] and STL_SUPPORT:
        return process_stl_file(cad_file)
    else:
        return None, {"error": f"未対応のCAD形式です: {file_ext}"}

def extract_text_with_ocr(image_file):
    """OCRでテキストを抽出"""
    if not OCR_SUPPORT:
        return ""
    
    try:
        image_file.seek(0)
        image = Image.open(image_file)
        image_file.seek(0)
        
        extracted_text = pytesseract.image_to_string(image, lang='jpn+eng')
        return extracted_text.strip() if extracted_text.strip() else ""
    except Exception as e:
        logger.error(f"OCRエラー: {e}")
        return ""

def analyze_image_with_gpt4o(image_base64, filename, cad_metadata=None, client=None):
    """GPT-4oで画像解析（CADメタデータ対応）"""
    if client is None:
        client = get_openai_client()
        if client is None:
            return {"error": "OpenAIクライアントが利用できません"}
    
    try:
        if cad_metadata:
            cad_info = f"""
CADファイル情報:
- ファイル形式: {cad_metadata.get('file_type', 'Unknown')}
- エンティティ数: {cad_metadata.get('total_entities', 'N/A')}
- 技術仕様: {cad_metadata}
"""
            prompt = f"""
この技術図面・CADファイル（ファイル名: {filename}）を詳細に分析し、以下の情報をJSON形式で返してください：

{cad_info}

1. image_type: 図面の種類（機械図面、建築図面、回路図、組織図、3Dモデル、その他）
2. main_content: 図面の主要な内容と技術的説明（300-400文字）
3. technical_specifications: 技術仕様・寸法・材質などの詳細情報
4. detected_elements: 図面内の主要な要素・部品リスト（最大15個）
5. dimensions_info: 寸法情報や測定値（検出できる場合）
6. annotations: 注記・文字情報・記号の内容
7. drawing_standards: 図面規格・標準（JIS、ISO、ANSI等、該当する場合）
8. manufacturing_info: 製造情報・加工情報（該当する場合）
9. keywords: 技術検索用キーワード（最大20個）
10. category_tags: 専門分野タグ（機械工学、建築、電気工学等、最大10個）
11. description_for_search: 技術者向け検索結果表示用説明（100-150文字）
12. related_standards: 関連する技術標準・規格の提案

JSON形式で返してください。技術的な観点から詳細に分析してください。
"""
        else:
            prompt = f"""
この画像（ファイル名: {filename}）を詳細に分析し、以下の情報をJSON形式で返してください：

1. image_type: 画像の種類（写真、技術図面、組織図、フローチャート、グラフ、表、地図、その他）
2. main_content: 画像の主要な内容の詳細説明（200-300文字）
3. detected_elements: 画像内の主要な要素リスト（最大10個）
4. technical_details: 技術的な詳細（寸法、規格、仕様など、該当する場合）
5. text_content: 画像内に含まれるテキスト内容（すべて正確に読み取って記載）
6. keywords: 検索に有用なキーワード（画像内容＋テキスト内容から最大20個）
7. search_terms: テキスト内容から想定される検索ワード・フレーズ（最大15個）
8. category_tags: 分類タグ（最大8個）
9. description_for_search: 検索結果表示用の簡潔な説明（80-120文字）
10. metadata_suggestions: 追加すべきメタデータの提案
11. related_topics: 画像・テキスト内容から関連しそうなトピック（最大10個）
12. document_type_hints: 文書種別の推定（報告書、マニュアル、仕様書、比較表等）

特に重要：
- text_contentには画像内のすべてのテキストを正確に読み取って記載してください
- そのテキスト内容を基に、検索で使われそうなキーワードやフレーズを多数生成してください
- 専門用語、固有名詞、数値、日付なども検索キーワードに含めてください

JSON形式で返してください。日本語で回答してください。
"""

        response = client.chat.completions.create(
            model=GPT4O_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=3000
        )
        
        result = json.loads(response.choices[0].message.content)
        
        if cad_metadata:
            result['cad_metadata'] = cad_metadata
            
        return result
        
    except Exception as e:
        logger.error(f"GPT-4o画像解析エラー: {e}")
        return {"error": f"画像解析中にエラーが発生しました: {e}"}

def get_embedding(text, client=None, dimensions=EMBEDDING_DIMENSIONS):
    """テキストの埋め込みベクトルを生成（text-embedding-3-large最適化）"""
    if client is None:
        client = get_openai_client()
        if client is None:
            return None
    
    try:
        if not text or not text.strip():
            return None
            
        # text-embedding-3-largeは8191トークンまで対応
        if len(text) > 30000:
            text = text[:30000]
        
        # dimensionsパラメータ対応（コスト効率化）
        params = {
            "model": EMBEDDING_MODEL,
            "input": text,
            "dimensions": dimensions
        }
            
        response = client.embeddings.create(**params)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"埋め込みベクトル生成エラー: {e}")
        return None

def create_comprehensive_search_chunk(analysis_result, user_additions):
    """★ ベクトル化用の包括的検索チャンクを作成"""
    chunk_parts = []
    
    # 基本情報
    if analysis_result.get('image_type'):
        chunk_parts.append(f"画像タイプ: {analysis_result['image_type']}")
    
    if analysis_result.get('main_content'):
        chunk_parts.append(f"主要内容: {analysis_result['main_content']}")
    
    # 検出要素
    elements = analysis_result.get('detected_elements', [])
    if elements:
        chunk_parts.append(f"主要要素: {', '.join(elements)}")
    
    # 技術詳細
    tech_details = analysis_result.get('technical_details', '')
    if tech_details:
        chunk_parts.append(f"技術詳細: {tech_details}")
    
    # 技術仕様（CAD用）
    tech_specs = analysis_result.get('technical_specifications', '')
    if tech_specs:
        chunk_parts.append(f"技術仕様: {tech_specs}")
    
    # 寸法情報
    dimensions_info = analysis_result.get('dimensions_info', '')
    if dimensions_info:
        chunk_parts.append(f"寸法情報: {dimensions_info}")
    
    # GPTが読み取ったテキスト内容（重要：検索対象）
    text_content = analysis_result.get('text_content', '')
    if text_content and text_content.strip():
        chunk_parts.append(f"画像内テキスト: {text_content}")
    
    # 注記・記号
    annotations = analysis_result.get('annotations', '')
    if annotations:
        chunk_parts.append(f"注記・記号: {annotations}")
    
    # ユーザー追加情報
    if user_additions.get('additional_description'):
        chunk_parts.append(f"補足説明: {user_additions['additional_description']}")
    
    if user_additions.get('purpose'):
        chunk_parts.append(f"用途・目的: {user_additions['purpose']}")
    
    if user_additions.get('context'):
        chunk_parts.append(f"文脈・背景: {user_additions['context']}")
    
    if user_additions.get('related_documents'):
        chunk_parts.append(f"関連文書: {user_additions['related_documents']}")
    
    # キーワード統合
    keywords = analysis_result.get('keywords', [])
    user_keywords = user_additions.get('additional_keywords', [])
    search_terms = analysis_result.get('search_terms', [])
    all_keywords = keywords + user_keywords + search_terms
    if all_keywords:
        chunk_parts.append(f"キーワード: {', '.join(set(all_keywords))}")  # 重複除去
    
    # 関連トピック
    related_topics = analysis_result.get('related_topics', [])
    if related_topics:
        chunk_parts.append(f"関連トピック: {', '.join(related_topics)}")
    
    return "\n".join(chunk_parts)

def create_structured_metadata(analysis_result, user_additions, filename):
    """★ 構造化メタデータを作成（検索結果表示用）"""
    return {
        # 基本情報
        "filename": filename,
        "image_type": analysis_result.get('image_type', ''),
        "category": user_additions.get('category', ''),
        "importance": user_additions.get('importance', '中'),
        
        # 表示用説明
        "title": user_additions.get('title', analysis_result.get('main_content', '')[:50] + '...'),
        "description_for_search": analysis_result.get('description_for_search', ''),
        "main_content": analysis_result.get('main_content', ''),
        
        # ユーザー情報
        "purpose": user_additions.get('purpose', ''),
        "context": user_additions.get('context', ''),
        "related_documents": user_additions.get('related_documents', ''),
        
        # 分類・タグ
        "keywords": analysis_result.get('keywords', []) + user_additions.get('additional_keywords', []),
        "search_terms": analysis_result.get('search_terms', []),
        "category_tags": analysis_result.get('category_tags', []),
        "related_topics": analysis_result.get('related_topics', []),
        
        # 技術情報
        "technical_details": analysis_result.get('technical_details', ''),
        "technical_specifications": analysis_result.get('technical_specifications', ''),
        "dimensions_info": analysis_result.get('dimensions_info', ''),
        "drawing_standards": analysis_result.get('drawing_standards', ''),
        "manufacturing_info": analysis_result.get('manufacturing_info', ''),
        
        # テキスト内容
        "text_content": analysis_result.get('text_content', ''),
        "annotations": analysis_result.get('annotations', ''),
        
        # 要素
        "detected_elements": analysis_result.get('detected_elements', []),
        
        # CADメタデータ
        "cad_metadata": analysis_result.get('cad_metadata', {})
    }

def save_unified_knowledge_item(image_id, analysis_result, user_additions, embedding, filename, image_base64=None, original_bytes=None):
    """★ 統一ナレッジアイテムとして保存（RAGシステム互換構造）"""
    try:
        search_chunk = create_comprehensive_search_chunk(analysis_result, user_additions)
        structured_metadata = create_structured_metadata(analysis_result, user_additions, filename)
        kb_name = "multimodal_knowledge_base"
        image_bytes = base64.b64decode(image_base64) if image_base64 else None

        full_metadata = {
            "filename": filename,
            "display_metadata": structured_metadata,
            "analysis_data": {
                "gpt_analysis": analysis_result,
                "cad_metadata": analysis_result.get('cad_metadata', {}),
                "user_additions": user_additions,
            },
        }

        paths = save_processed_data(
            kb_name,
            image_id,
            chunk_text=search_chunk,
            embedding=embedding,
            metadata=full_metadata,
            original_filename=filename,
            original_bytes=original_bytes,
            image_bytes=image_bytes,
        )
        refresh_search_engine(kb_name)
        file_link = paths.get("original_file_path", "")

        return True, {
            "id": image_id,
            "type": "image",
            "filename": filename,
            "chunk_path": paths.get("chunk_path"),
            "embedding_path": paths.get("embedding_path"),
            "metadata_path": paths.get("metadata_path"),
            "image_path": paths.get("image_path"),
            "file_link": file_link,
        }
        
    except Exception as e:
        logger.error(f"ナレッジデータ保存エラー: {e}")
        return False, None

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# メインUI
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

st.title("◊ マルチモーダルナレッジ構築ツール")
st.markdown("画像・図面からナレッジベースを構築するための改良版ツールです。")

# サイドバー設定
st.sidebar.header("⚡ 設定")
show_debug = st.sidebar.checkbox("デバッグ情報を表示", value=False)
embedding_dims = st.sidebar.selectbox(
    "埋め込み次元数",
    [1536, 3072],
    index=0,
    help="1536: コスト効率重視、3072: 精度重視"
)

# メインタブ
tab1, tab2, tab3 = st.tabs(["↑ 画像アップロード", "∠ 内容編集・ナレッジ化", "≡ ナレッジベース管理"])

with tab1:
    st.header("画像・図面のアップロード")
    
    # ファイル形式の説明
    with st.expander("⟐ 対応ファイル形式"):
        st.markdown(f"""
        **画像ファイル**: {', '.join(SUPPORTED_IMAGE_TYPES)}
        **文書ファイル**: {', '.join(SUPPORTED_DOCUMENT_TYPES) if PDF_SUPPORT else 'PDF処理無効'}
        **CADファイル**: {', '.join(SUPPORTED_CAD_TYPES)}
        
        ### ⚙ CAD形式対応状況
        - **DXF** ◎ - AutoCAD図面交換形式 {('(対応済み)' if DXF_SUPPORT else '(要ezdxf)')}
        - **STL** ◎ - 3Dプリンタ用メッシュファイル {('(対応済み)' if STL_SUPPORT else '(要trimesh)')}
        - **STEP/STP** ◇ - 3D CAD標準交換形式 {('(対応済み)' if STEP_SUPPORT else '(要cadquery)')}
        - **PLY/OBJ** ◇ - 3Dメッシュファイル {('(対応済み)' if STL_SUPPORT else '(要trimesh)')}
        """)
    
    # ファイルアップロード
    uploaded_files = st.file_uploader(
        "画像・CADファイルを選択してください",
        type=SUPPORTED_IMAGE_TYPES + SUPPORTED_DOCUMENT_TYPES + SUPPORTED_CAD_TYPES,
        accept_multiple_files=True,
        help="複数ファイルの同時アップロードが可能です。CADファイルは自動的に画像に変換されます。"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)}個のファイルがアップロードされました")
        
        if st.button("⌕ AI解析を開始", type="primary"):
            client = get_openai_client()
            if not client:
                st.error("OpenAIクライアントに接続できません")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    file_bytes = uploaded_file.getvalue()
                    uploaded_file.seek(0)
                    status_text.text(f"処理中: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                    
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    is_cad_file = file_extension in SUPPORTED_CAD_TYPES
                    
                    image_base64 = None
                    cad_metadata = None
                    
                    if is_cad_file:
                        with st.spinner(f"CADファイル処理中: {uploaded_file.name}"):
                            image_base64, cad_metadata = process_cad_file(uploaded_file, file_extension)
                            if image_base64 is None:
                                st.error(f"CADファイル処理エラー: {uploaded_file.name} - {cad_metadata.get('error', '不明なエラー')}")
                                continue
                    else:
                        image_base64 = encode_image_to_base64(uploaded_file)
                        if not image_base64:
                            st.error(f"ファイル処理エラー: {uploaded_file.name}")
                            continue
                    
                    # GPT-4o解析
                    with st.spinner(f"GPT-4.1で解析中: {uploaded_file.name}"):
                        analysis = analyze_image_with_gpt4o(image_base64, uploaded_file.name, cad_metadata, client)
                    
                    if "error" not in analysis:
                        image_id = str(uuid.uuid4())
                        st.session_state.processed_images[image_id] = {
                            'filename': uploaded_file.name,
                            'file_extension': file_extension,
                            'is_cad_file': is_cad_file,
                            'image_base64': image_base64,
                            'analysis': analysis,
                            'cad_metadata': cad_metadata,
                            'user_additions': {},
                            'is_finalized': False,
                            'original_bytes': file_bytes,
                        }
                        
                        file_type_display = "CADファイル" if is_cad_file else "画像"
                        st.success(f"◎ {uploaded_file.name} ({file_type_display}) の解析完了")
                    else:
                        st.error(f"× {uploaded_file.name} の解析失敗: {analysis.get('error')}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("全ての処理が完了しました")

with tab2:
    st.header("内容編集・ナレッジ化")
    
    if not st.session_state.processed_images:
        st.info("「画像アップロード」タブで画像を処理してください")
    else:
        # 画像選択
        image_options = {f"{data['filename']} (ID: {img_id[:8]}...)": img_id 
                        for img_id, data in st.session_state.processed_images.items()}
        
        selected_display = st.selectbox(
            "編集する画像を選択",
            list(image_options.keys()),
            index=0
        )
        
        if selected_display:
            selected_id = image_options[selected_display]
            image_data = st.session_state.processed_images[selected_id]
            
            # ★★★ 3列レイアウト：画像、AIデータ、ユーザー編集 ★★★
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.subheader("◊ 画像プレビュー")
                
                try:
                    image_bytes = base64.b64decode(image_data['image_base64'])
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption=image_data['filename'], use_container_width=True)
                except Exception as e:
                    st.error(f"画像表示エラー: {e}")
                
                # ファイル情報
                if image_data.get('is_cad_file', False):
                    st.info(f"⚙ CADファイル: {image_data['file_extension'].upper()}")
                    
                    if image_data.get('cad_metadata'):
                        cad_meta = image_data['cad_metadata']
                        with st.expander("⚙ CAD技術情報"):
                            if cad_meta.get('file_type'):
                                st.write(f"**形式**: {cad_meta['file_type']}")
                            if cad_meta.get('total_entities'):
                                st.write(f"**エンティティ数**: {cad_meta['total_entities']}")
                            if cad_meta.get('vertices_count'):
                                st.write(f"**頂点数**: {cad_meta['vertices_count']}")
                            if cad_meta.get('volume'):
                                st.write(f"**体積**: {cad_meta['volume']:.3f}")
            
            with col2:
                st.subheader("∷ AI解析結果")
                analysis = image_data['analysis']
                
                st.write(f"**画像タイプ**: {analysis.get('image_type', 'N/A')}")
                
                with st.expander("∠ AI抽出内容（参考）", expanded=False):
                    if analysis.get('main_content'):
                        st.write(f"**主要内容**: {analysis['main_content']}")
                    
                    if analysis.get('detected_elements'):
                        st.write("**検出要素**:")
                        for element in analysis['detected_elements'][:5]:  # 上位5つのみ表示
                            st.write(f"- {element}")
                    
                    if analysis.get('text_content'):
                        st.write(f"**画像内テキスト**: {analysis['text_content'][:200]}...")
                    
                    if analysis.get('keywords'):
                        st.write(f"**AI生成キーワード**: {', '.join(analysis['keywords'][:8])}...")
                    
                    if analysis.get('technical_details'):
                        st.write(f"**技術詳細**: {analysis['technical_details']}")
                
                # ★★★ リアルタイムプレビュー ★★★
                st.subheader("⌕ ナレッジプレビュー")
                
                if st.button("⟐ 最新プレビューを生成"):
                    user_additions = image_data.get('user_additions', {})
                    
                    # プレビューチャンク作成
                    preview_chunk = create_comprehensive_search_chunk(analysis, user_additions)
                    preview_metadata = create_structured_metadata(analysis, user_additions, image_data['filename'])
                    
                    st.markdown("**□ ベクトル化される検索チャンク:**")
                    with st.container():
                        st.text_area("", preview_chunk, height=150, disabled=True, key="preview_chunk")
                        st.caption(f"文字数: {len(preview_chunk)}")
                    
                    st.markdown("**≡ 構造化メタデータ（検索結果表示用）:**")
                    with st.expander("メタデータ詳細", expanded=False):
                        st.json(preview_metadata)
            
            with col3:
                st.subheader("∠ ユーザー追加情報")
                
                user_additions = image_data.get('user_additions', {})
                
                # タイトル
                title = st.text_input(
                    "∠ タイトル",
                    value=user_additions.get('title', ''),
                    help="検索結果に表示されるタイトル"
                )
                
                # 補足説明
                additional_description = st.text_area(
                    "⟐ 補足説明",
                    value=user_additions.get('additional_description', ''),
                    help="AIの解析に追加したい詳細な説明（重要：検索対象になります）",
                    height=100
                )
                
                # 用途・目的
                purpose = st.text_input(
                    "◉ 用途・目的",
                    value=user_additions.get('purpose', ''),
                    help="この画像の用途や目的"
                )
                
                # 文脈・背景
                context = st.text_area(
                    "≈ 文脈・背景",
                    value=user_additions.get('context', ''),
                    help="この画像の背景情報や文脈",
                    height=80
                )
                
                # 関連文書
                related_documents = st.text_input(
                    "⟐ 関連文書",
                    value=user_additions.get('related_documents', ''),
                    help="関連する文書やファイル名"
                )
                
                # 追加キーワード
                additional_keywords_str = st.text_input(
                    "⊞ 追加キーワード（カンマ区切り）",
                    value=', '.join(user_additions.get('additional_keywords', [])),
                    help="検索用の追加キーワード（重要：検索性能向上）"
                )
                additional_keywords = [kw.strip() for kw in additional_keywords_str.split(',') if kw.strip()]
                
                # カテゴリと重要度
                col3_1, col3_2 = st.columns(2)
                with col3_1:
                    category_options = ["技術文書", "組織図", "フローチャート", "データ図表", "写真", "地図", "その他"]
                    selected_category = st.selectbox(
                        "≣ カテゴリ",
                        category_options,
                        index=category_options.index(user_additions.get('category', '技術文書')) 
                              if user_additions.get('category') in category_options else 0
                    )
                
                with col3_2:
                    importance = st.select_slider(
                        "◇ 重要度",
                        options=["低", "中", "高", "最重要"],
                        value=user_additions.get('importance', '中')
                    )
                
                # 情報更新
                if st.button("□ 情報を更新", type="secondary"):
                    st.session_state.processed_images[selected_id]['user_additions'] = {
                        'title': title,
                        'additional_description': additional_description,
                        'purpose': purpose,
                        'context': context,
                        'related_documents': related_documents,
                        'additional_keywords': additional_keywords,
                        'category': selected_category,
                        'importance': importance
                    }
                    st.success("◎ 情報が更新されました")
                    st.rerun()
                
                st.markdown("---")
                
                # ★★★ ナレッジベース登録 ★★★
                st.subheader("⤴ ナレッジベース登録")
                
                if not image_data.get('is_finalized', False):
                    if st.button("◎ ナレッジベースに登録", type="primary"):
                        with st.spinner("ナレッジベース登録中..."):
                            client = get_openai_client()
                            if client:
                                # 最新のユーザー情報を取得
                                current_user_additions = st.session_state.processed_images[selected_id]['user_additions']
                                
                                # チャンク作成
                                search_chunk = create_comprehensive_search_chunk(analysis, current_user_additions)
                                
                                # 埋め込みベクトル生成
                                embedding = get_embedding(search_chunk, client, dimensions=embedding_dims)
                                
                                if embedding:
                                    # 統一ナレッジアイテムとして保存
                                    success, saved_item = save_unified_knowledge_item(
                                        selected_id,
                                        analysis,
                                        current_user_additions,
                                        embedding,
                                        image_data['filename'],
                                        image_data['image_base64'],
                                        original_bytes=image_data.get('original_bytes')
                                    )
                                    
                                    if success:
                                        st.session_state.processed_images[selected_id]['is_finalized'] = True
                                        st.success("◎ ナレッジベースに登録完了！")
                                        
                                        # 登録結果表示（分離構造対応）
                                        with st.expander("≡ 登録されたデータ（既存RAGシステム互換）", expanded=True):
                                            st.write(f"**ID**: {saved_item['id']}")
                                            st.write(f"**ファイルリンク**: {saved_item['file_link']}")
                                            st.write(f"**ベクトル次元数**: {saved_item['stats']['vector_dimensions']}")
                                            st.write(f"**キーワード数**: {saved_item['stats']['keywords_count']}")
                                            st.write(f"**チャンク文字数**: {saved_item['stats']['chunk_length']}")
                                            
                                            st.markdown("**⟐ 保存先ファイル:**")
                                            st.code(f"""
chunks/{saved_item['id']}.json      # 検索用テキストチャンク
embeddings/{saved_item['id']}.json  # ベクトルデータ
metadata/{saved_item['id']}.json    # メタ情報
images/{saved_item['id']}.jpg       # 画像ファイル
files/{saved_item['id']}_info.json  # ファイル情報
                                            """)
                                        
                                        if show_debug:
                                            with st.expander("⚙ デバッグ: ベクトル化チャンク"):
                                                st.text_area("", search_chunk, height=150, disabled=True)
                                    else:
                                        st.error("× ナレッジベース登録中にエラーが発生しました")
                                else:
                                    st.error("× ベクトル化に失敗しました")
                            else:
                                st.error("× OpenAIクライアントに接続できません")
                else:
                    st.success("◎ この画像は既にナレッジベース登録済みです")
                    
                    if st.button("⟲ 再登録（情報更新）", help="情報を更新して再度登録します"):
                        st.session_state.processed_images[selected_id]['is_finalized'] = False
                        st.rerun()

with tab3:
    st.header("ナレッジベース管理")
    
    # ナレッジベース選択
    kb_name = st.selectbox(
        "ナレッジベース選択",
        ["multimodal_knowledge_base"],  # 将来複数KB対応可能
        index=0
    )
    
    # 既存RAGシステム互換のディレクトリ構造
    kb_dir = DATA_DIR / kb_name
    chunks_dir = kb_dir / "chunks"
    embeddings_dir = kb_dir / "embeddings"
    metadata_dir = kb_dir / "metadata"
    images_dir = kb_dir / "images"
    files_dir = kb_dir / "files"
    
    if metadata_dir.exists():
        metadata_files = list(metadata_dir.glob("*.json"))
        if metadata_files:
            # ナレッジベース統計表示
            kb_metadata_path = kb_dir / "kb_metadata.json"
            if kb_metadata_path.exists():
                try:
                    with open(kb_metadata_path, 'r', encoding='utf-8') as f:
                        kb_info = json.load(f)
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("≡ 総アイテム数", kb_info.get('total_items', len(metadata_files)))
                    with col_stat2:
                        st.metric("◊ 画像", kb_info.get('item_types', {}).get('image', len(metadata_files)))
                    with col_stat3:
                        st.metric("⟐ テキスト", kb_info.get('item_types', {}).get('text_chunk', 0))
                    with col_stat4:
                        st.metric("⟲ 最終更新", kb_info.get('last_updated', '')[:10] if kb_info.get('last_updated') else 'N/A')
                except:
                    st.info(f"≡ ナレッジベース登録データ: {len(metadata_files)}件")
            else:
                st.info(f"≡ ナレッジベース登録データ: {len(metadata_files)}件")
            
            # ディレクトリ構造表示
            with st.expander("⟐ ディレクトリ構造（既存RAGシステム互換）", expanded=False):
                st.code(f"""
⟐ {kb_name}/
├── ⟐ chunks/      ({len(list(chunks_dir.glob('*.json')) if chunks_dir.exists() else [])}件)
├── ≈ embeddings/  ({len(list(embeddings_dir.glob('*.json')) if embeddings_dir.exists() else [])}件)
├── ≡ metadata/    ({len(list(metadata_dir.glob('*.json')) if metadata_dir.exists() else [])}件)
├── ◊ images/      ({len(list(images_dir.glob('*.*')) if images_dir.exists() else [])}件)
├── ⟐ files/       ({len(list(files_dir.glob('*.json')) if files_dir.exists() else [])}件)
└── ⟐ kb_metadata.json
                """)
            
            # データ一覧表示
            data_list = []
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    display_meta = data.get('display_metadata', {})
                    stats = data.get('stats', {})
                    
                    data_list.append({
                        'ID': data['id'][:8] + '...',
                        'ファイル名': data.get('filename', 'N/A'),
                        'タイトル': display_meta.get('title', 'N/A')[:30] + '...' if len(display_meta.get('title', '')) > 30 else display_meta.get('title', 'N/A'),
                        '画像タイプ': display_meta.get('image_type', 'N/A'),
                        'カテゴリ': display_meta.get('category', 'N/A'),
                        '重要度': display_meta.get('importance', 'N/A'),
                        'キーワード数': stats.get('keywords_count', 0),
                        'チャンク文字数': stats.get('chunk_length', 0),
                        'ベクトル次元': stats.get('vector_dimensions', 0),
                        'ファイルリンク': data.get('file_link', 'N/A'),
                        '作成日時': data.get('created_at', '')[:19]
                    })
                except Exception as e:
                    logger.error(f"メタデータ読み込みエラー {metadata_file}: {e}")
            
            if data_list:
                df = pd.DataFrame(data_list)
                st.dataframe(df, use_container_width=True)
                
                # エクスポート機能
                if st.button("↓ ナレッジベースをCSVエクスポート"):
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="□ CSVダウンロード",
                        data=csv,
                        file_name=f"{kb_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # ★★★ 改良版ナレッジ検索（分離構造対応） ★★★
                st.subheader("⌕ ナレッジ検索")
                
                col_search1, col_search2 = st.columns([3, 1])
                with col_search1:
                    search_query = st.text_input("検索クエリを入力", placeholder="例: WEB版 比較項目、技術仕様、組織図")
                
                with col_search2:
                    search_top_k = st.selectbox("表示件数", [5, 10, 15, 20], index=1)
                
                if search_query and st.button("⌕ 検索実行", type="primary"):
                    client = get_openai_client()
                    if client:
                        with st.spinner("検索中..."):
                            # クエリのベクトル化
                            query_embedding = get_embedding(search_query, client, dimensions=embedding_dims)
                            
                            if query_embedding:
                                # 類似度計算（分離構造対応）
                                similarities = []
                                
                                for metadata_file in metadata_files:
                                    try:
                                        # メタデータ読み込み
                                        with open(metadata_file, 'r', encoding='utf-8') as f:
                                            metadata = json.load(f)
                                        
                                        item_id = metadata['id']
                                        
                                        # 対応するベクトルファイル読み込み
                                        embedding_file = embeddings_dir / f"{item_id}.json"
                                        if embedding_file.exists():
                                            with open(embedding_file, 'r', encoding='utf-8') as f:
                                                embedding_data = json.load(f)
                                            
                                            doc_embedding = embedding_data.get('vector')
                                            if doc_embedding and len(doc_embedding) == len(query_embedding):
                                                # コサイン類似度計算
                                                similarity = np.dot(query_embedding, doc_embedding) / (
                                                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                                                )
                                                
                                                # チャンクデータも読み込み（検索結果表示用）
                                                chunk_file = chunks_dir / f"{item_id}.json"
                                                chunk_data = {}
                                                if chunk_file.exists():
                                                    with open(chunk_file, 'r', encoding='utf-8') as f:
                                                        chunk_data = json.load(f)
                                                
                                                similarities.append({
                                                    'metadata': metadata,
                                                    'chunk_data': chunk_data,
                                                    'embedding_data': embedding_data,
                                                    'similarity': similarity
                                                })
                                    except Exception as e:
                                        logger.error(f"検索エラー {metadata_file}: {e}")
                                
                                # 類似度順でソート
                                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                                
                                # 検索結果表示
                                st.write(f"**◉ 検索結果（上位{min(search_top_k, len(similarities))}件）**")
                                
                                if similarities:
                                    for i, result in enumerate(similarities[:search_top_k]):
                                        metadata = result['metadata']
                                        display_meta = metadata.get('display_metadata', {})
                                        stats = metadata.get('stats', {})
                                        chunk_data = result.get('chunk_data', {})
                                        
                                        with st.expander(f"{i+1}. {display_meta.get('title', metadata.get('filename', 'N/A'))} (類似度: {result['similarity']:.3f})"):
                                            # 4列レイアウト
                                            result_col1, result_col2, result_col3, result_col4 = st.columns([1, 1, 1, 1])
                                            
                                            with result_col1:
                                                st.markdown("**⟐ 基本情報**")
                                                st.write(f"**ファイル名**: {metadata.get('filename', 'N/A')}")
                                                st.write(f"**タイプ**: {display_meta.get('image_type', 'N/A')}")
                                                st.write(f"**カテゴリ**: {display_meta.get('category', 'N/A')}")
                                                st.write(f"**重要度**: {display_meta.get('importance', 'N/A')}")
                                                
                                                # ★ ファイルリンク表示
                                                file_link = metadata.get('file_link', '')
                                                if file_link:
                                                    st.write(f"**⟐ ファイルリンク**: `{file_link}`")
                                            
                                            with result_col2:
                                                st.markdown("**≡ 統計・スコア**")
                                                st.metric("類似度", f"{result['similarity']:.3f}")
                                                st.write(f"**キーワード数**: {stats.get('keywords_count', 0)}")
                                                st.write(f"**チャンク文字数**: {stats.get('chunk_length', 0)}")
                                                st.write(f"**ベクトル次元**: {stats.get('vector_dimensions', 0)}")
                                                st.write(f"**作成日**: {metadata.get('created_at', '')[:10]}")
                                            
                                            with result_col3:
                                                st.markdown("**◊ 画像プレビュー**")
                                                try:
                                                    item_id = metadata['id']
                                                    image_file = images_dir / f"{item_id}.jpg"
                                                    if image_file.exists():
                                                        image = Image.open(image_file)
                                                        st.image(image, width=150)
                                                    else:
                                                        st.write("画像ファイルなし")
                                                except Exception as e:
                                                    st.write("画像プレビュー不可")
                                            
                                            with result_col4:
                                                st.markdown("**⟐ ファイル構造**")
                                                item_id = metadata['id']
                                                st.write(f"**チャンク**: `chunks/{item_id}.json`")
                                                st.write(f"**ベクトル**: `embeddings/{item_id}.json`")
                                                st.write(f"**メタデータ**: `metadata/{item_id}.json`")
                                                st.write(f"**画像**: `images/{item_id}.jpg`")
                                            
                                            # 詳細情報
                                            if display_meta.get('main_content'):
                                                st.markdown("**∠ 内容**")
                                                content = display_meta['main_content']
                                                st.write(content[:200] + '...' if len(content) > 200 else content)
                                            
                                            if display_meta.get('keywords'):
                                                st.markdown("**⊞ キーワード**")
                                                keywords = display_meta['keywords']
                                                keywords_display = ', '.join(keywords[:10])
                                                if len(keywords) > 10:
                                                    keywords_display += f" （他{len(keywords)-10}個）"
                                                st.write(keywords_display)
                                            
                                            if display_meta.get('purpose'):
                                                st.markdown("**◉ 用途**")
                                                st.write(display_meta['purpose'])
                                            
                                            # デバッグ情報
                                            if show_debug:
                                                with st.expander("⚙ デバッグ: チャンクとファイルパス"):
                                                    st.markdown("**検索チャンク:**")
                                                    st.text_area("", chunk_data.get('content', ''), height=100, disabled=True, key=f"debug_chunk_{i}")
                                                    st.markdown("**ファイルパス:**")
                                                    st.write(f"chunks: {chunks_dir / f'{item_id}.json'}")
                                                    st.write(f"embeddings: {embeddings_dir / f'{item_id}.json'}")
                                                    st.write(f"metadata: {metadata_dir / f'{item_id}.json'}")
                                else:
                                    st.info("検索結果が見つかりませんでした。検索語を変更してみてください。")
                            else:
                                st.error("× 検索クエリのベクトル化に失敗しました")
                    else:
                        st.error("× OpenAIクライアントに接続できません")
        else:
            st.info("ナレッジベースにデータがありません")
    else:
        st.info(f"ナレッジベース '{kb_name}' が見つかりません")
    
    # 現在のセッション統計
    if st.session_state.processed_images:
        st.subheader("≈ 現在のセッション統計")
        
        total_images = len(st.session_state.processed_images)
        finalized_images = sum(1 for data in st.session_state.processed_images.values() 
                              if data.get('is_finalized', False))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("処理済み画像", total_images)
        with col2:
            st.metric("登録済み", finalized_images)
        with col3:
            st.metric("未登録", total_images - finalized_images)
        with col4:
            st.metric("ベクトル次元", embedding_dims)

# フッター
st.markdown("---")
st.markdown(
    '<div class="footer-text">マルチモーダルナレッジ構築ツール v3.0.0 - 既存RAGシステム統合対応版</div>', 
    unsafe_allow_html=True
)