import os
import json
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer # BM25には不要
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sentence_transformers import SentenceTransformer
# import time # 直接は不要
from rank_bm25 import BM25Okapi
import nltk
# from nltk.tokenize import word_tokenize # SudachiPyに置き換え、またはフォールバックとして保持
from nltk.corpus import stopwords
from pathlib import Path
import traceback
import re
import typing # ★ typing モジュールをインポート

# --- SudachiPyのインポートと初期化 ---
try:
    from sudachipy import tokenizer as sudachi_tokenizer_module
    from sudachipy import dictionary as sudachi_dictionary_module
    _sudachi_tokenizer_instance_for_bm25 = sudachi_dictionary_module.Dictionary().create()
    _SUDASHI_BM25_TOKENIZER_MODE = sudachi_tokenizer_module.Tokenizer.SplitMode.B
    print("SudachiPy tokenizer for BM25 (knowledge_search.py) initialized successfully.")
except ImportError:
    print("WARNING (knowledge_search.py): SudachiPy not found. BM25 will use a fallback regex tokenizer, which is not ideal for Japanese.")
    _sudachi_tokenizer_instance_for_bm25 = None
except Exception as e_sudachi_init:
    print(f"WARNING (knowledge_search.py): SudachiPy tokenizer for BM25 failed to initialize: {e_sudachi_init}. Falling back to regex.")
    _sudachi_tokenizer_instance_for_bm25 = None
# --- SudachiPyのインポートと初期化ここまで ---

# NLTKリソースダウンロード関数
def ensure_nltk_resources():
    """必要なNLTKリソースが確実にダウンロードされるようにする"""
    try:
        print("NLTKリソースの確認とダウンロードを開始...")
        resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                print(f"リソース '{resource}' は既にダウンロード済みです")
            except LookupError:
                print(f"リソース '{resource}' をダウンロード中...")
                nltk.download(resource, quiet=True)
                print(f"リソース '{resource}' のダウンロードが完了しました")
        return True
    except Exception as e:
        print(f"NLTKリソースのダウンロード中にエラーが発生しました: {e}")
        return False

ensure_nltk_resources()

_stop_words_set = set()
try:
    _stop_words_set.update(stopwords.words('english'))
    _japanese_stopwords_list = [
        'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ', 'ある', 'いる', 'から', 'など', 'なっ', 'ない', 'ので',
        'です', 'ます', 'する', 'もの', 'こと', 'よう', 'ため', 'において', 'における', 'および', 'また', 'も', 'という', 'られる',
        'により', 'に関する', 'ついて', 'として', ' terhadap', 'によって', 'より', 'における', 'に関する', 'に対する', 'としての',
        'あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す', 'せ', 'そ',
        'な', 'ぬ', 'ね', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ',
        'ら', 'り', 'る', 'ろ', 'わ', 'を', 'ん',
        'これ', 'それ', 'あれ', 'この', 'その', 'あの', 'ここ', 'そこ', 'あそこ', 'こちら', 'そちら', 'あちら',
        '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
        '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '、', '。', '「', '」', '（', '）', '・'
    ]
    _stop_words_set.update(_japanese_stopwords_list)
    print(f"ストップワードの初期化完了 (knowledge_search.py): {len(_stop_words_set)}語")
except Exception as e_stopwords:
    print(f"ストップワード初期化エラー (knowledge_search.py): {e_stopwords}")

def tokenize_text_for_bm25_internal(text_input: str) -> list[str]:
    if not isinstance(text_input, str) or not text_input.strip():
        return ["<bm25_empty_input_token>"]
    processed_text = text_input.lower()
    tokens: list[str] = []
    if _sudachi_tokenizer_instance_for_bm25:
        try:
            tokens = [
                m.normalized_form() 
                for m in _sudachi_tokenizer_instance_for_bm25.tokenize(processed_text, _SUDASHI_BM25_TOKENIZER_MODE)
            ]
        except Exception as e_sudachi_tokenize:
            print(f"    [Tokenizer] SudachiPyでのトークン化中にエラー: {e_sudachi_tokenize}. Regexフォールバック使用。Text: {processed_text[:30]}...")
            tokens = re.findall(r'[ぁ-んァ-ン一-龥a-zA-Z0-9]+', processed_text)
    else:
        tokens = re.findall(r'[ぁ-んァ-ン一-龥a-zA-Z0-9]+', processed_text)
    if not tokens:
        return ["<bm25_empty_tokenization_result>"]
    if _stop_words_set:
        tokens_after_stopwords = [token for token in tokens if token not in _stop_words_set]
        if not tokens_after_stopwords and tokens:
            return ["<bm25_all_stopwords_token>"]
        tokens = tokens_after_stopwords
    if not tokens:
        return ["<bm25_empty_after_stopwords_token>"]
    return tokens

class HybridSearchEngine:
    def __init__(self, kb_path: str):
        print(f"HybridSearchEngine初期化: kb_path={kb_path}")
        self.kb_path = Path(kb_path)
        self.chunks_path = self.kb_path / "chunks"
        self.metadata_path = self.kb_path / "metadata"
        self.embeddings_path = self.kb_path / "embeddings"
        
        self.bm25_index_file_path = self.kb_path / "bm25_index_sudachi.pkl"
        self.tokenized_corpus_file_path = self.kb_path / "tokenized_corpus_sudachi.pkl"

        print(f"パス確認: chunks={self.chunks_path.exists()}, metadata={self.metadata_path.exists()}, embeddings={self.embeddings_path.exists()}")
        
        self.kb_metadata = self._load_kb_metadata()
        self.embedding_model = self.kb_metadata.get('embedding_model', 'text-embedding-3-small')
        print(f"使用する埋め込みモデル: {self.embedding_model}")
        
        self.chunks = self._load_chunks()
        print(f"初期チャンク読み込み完了: {len(self.chunks)}件")

        self.embeddings = self._load_embeddings()
        print(f"埋め込みベクトル読み込み完了: {len(self.embeddings)}件")

        self._integrate_faq_chunks()
        
        self._check_chunk_embedding_consistency()
        
        self.tokenized_corpus_for_bm25: typing.Union[list[list[str]], None] = None # ★修正
        self.bm25_index: typing.Union[BM25Okapi, None] = None # ★修正
        self.bm25_index = self._load_or_build_bm25_index()
        print(f"BM25処理後の有効チャンク数: {len(self.chunks)}")
        
        try:
            print("バックアップ埋め込みモデル SentenceTransformer を読み込み中...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("SentenceTransformer 読み込み完了")
        except Exception as e_st:
            print(f"SentenceTransformer 読み込みエラー: {e_st}")
            self.model = None

    def reindex(self) -> None:
        """Reload all chunks and embeddings and rebuild the BM25 index."""
        self.chunks = self._load_chunks()
        self.embeddings = self._load_embeddings()
        self._integrate_faq_chunks()
        self._check_chunk_embedding_consistency()
        self.bm25_index = self._load_or_build_bm25_index()

    def _load_kb_metadata(self) -> dict:
        metadata_file = self.kb_path / "kb_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f: return json.load(f)
            except Exception as e: print(f"ナレッジベースメタデータ読み込みエラー: {e}"); traceback.print_exc()
        else:
            print(f"ナレッジベースメタデータファイルが見つかりません: {metadata_file}")
        return {}

    def _load_chunks(self) -> list[dict]:
        print("チャンクを読み込み中...")
        loaded_chunks = []
        if not self.chunks_path.exists():
            print(f"チャンクディレクトリが見つかりません: {self.chunks_path}")
            return []
            
        for chunk_file_path in self.chunks_path.glob("*.txt"):
            try:
                stem = chunk_file_path.stem
                match = re.search(r'(?:.*_)?(\d+)$', stem)
                if match:
                    chunk_id = match.group(1)
                else:
                    chunk_id = stem 
                    print(f"    警告: チャンクファイル名 {chunk_file_path.name} から標準的なIDを抽出できませんでした。StemをIDとして使用: {chunk_id}")
                with open(chunk_file_path, 'r', encoding='utf-8') as f:
                    chunk_text = f.read()
                metadata = {}
                metadata_file = self.metadata_path / f"metadata_{chunk_id}.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f: metadata = json.load(f)
                loaded_chunks.append({'id': chunk_id, 'text': chunk_text, 'metadata': metadata})
            except Exception as e: print(f"チャンク '{chunk_file_path.name}' の読み込み中にエラー: {e}"); traceback.print_exc()
        return loaded_chunks

    def _load_embeddings(self) -> dict[str, list[float]]:
        print("埋め込みベクトルを読み込み中...")
        loaded_embeddings = {}
        if not self.embeddings_path.exists():
            print(f"埋め込みディレクトリが見つかりません: {self.embeddings_path}")
            return {}
        for emb_file_path in self.embeddings_path.glob("*.pkl"):
            try:
                stem = emb_file_path.stem
                match = re.search(r'(?:.*_)?(\d+)$', stem)
                if match:
                    chunk_id = match.group(1)
                else:
                    chunk_id = stem
                    print(f"    警告: 埋込ファイル名 {emb_file_path.name} からID抽出失敗。StemをIDとして使用: {chunk_id}")
                with open(emb_file_path, 'rb') as f: embedding_data = pickle.load(f)
                emb_vector = None
                if isinstance(embedding_data, dict) and 'embedding' in embedding_data:
                    emb_vector = embedding_data['embedding']
                elif isinstance(embedding_data, (list, np.ndarray)):
                    emb_vector = embedding_data
                if emb_vector is not None:
                    loaded_embeddings[chunk_id] = np.array(emb_vector, dtype=np.float32).tolist()
            except Exception as e: print(f"埋め込みファイル '{emb_file_path.name}' の読み込み中にエラー: {e}")
        return loaded_embeddings
    
    def _check_chunk_embedding_consistency(self):
        chunk_ids = set(c['id'] for c in self.chunks)
        embedding_ids = set(self.embeddings.keys())
        print(f"整合性チェック: チャンクID数={len(chunk_ids)}, 埋め込みID数={len(embedding_ids)}")
        missing_embeddings_for_chunks = chunk_ids - embedding_ids
        if missing_embeddings_for_chunks:
            print(f"  警告: 次のチャンクIDには対応する埋め込みがありません (上位5件): {list(missing_embeddings_for_chunks)[:5]}")
        missing_chunks_for_embeddings = embedding_ids - chunk_ids
        if missing_chunks_for_embeddings:
            print(f"  警告: 次の埋め込みIDには対応するチャンクがありません (上位5件): {list(missing_chunks_for_embeddings)[:5]}")
        common_ids_count = len(chunk_ids.intersection(embedding_ids))
        print(f"  チャンクと埋め込みで共通のID数: {common_ids_count}")

    def _integrate_faq_chunks(self) -> None:
        """Load FAQ entries and append them to self.chunks."""
        faq_file = self.kb_path / "faqs.json"
        if not faq_file.exists():
            return
        try:
            with open(faq_file, "r", encoding="utf-8") as f:
                faqs = json.load(f)
        except Exception as e:
            print(f"FAQ読み込みエラー: {e}")
            return
        for faq in faqs:
            fid = faq.get("id")
            if not fid:
                continue
            text = f"Q: {faq.get('question', '')}\nA: {faq.get('answer', '')}"
            self.chunks.append({
                'id': fid,
                'text': text,
                'metadata': {'faq': True, 'question': faq.get('question'), 'answer': faq.get('answer')}
            })

    def _create_tokenized_corpus_and_filter_chunks(self) -> tuple[list[list[str]], list[dict]]:
        print("BM25用コーパスのトークン化とチャンクフィルタリングを開始...")
        tokenized_corpus: list[list[str]] = []
        successfully_processed_chunks: list[dict] = []
        if not self.chunks:
            print("  警告: self.chunksが空のため、トークン化できません。")
            return [], []
        for i, chunk_data in enumerate(self.chunks):
            chunk_text = chunk_data.get('text', '')
            tokens = tokenize_text_for_bm25_internal(chunk_text)
            is_dummy_token_only = all(token.startswith("<bm25_") and token.endswith("_token>") for token in tokens)
            if tokens and not is_dummy_token_only:
                tokenized_corpus.append(tokens)
                successfully_processed_chunks.append(chunk_data)
                if i % 100 == 0 and i > 0:
                    print(f"    ... {i}件のチャンクをトークン化済み ...")
            else:
                print(f"    警告: チャンクID {chunk_data.get('id', 'N/A')} のトークン化結果がBM25に不適格。除外します。Tokens: {tokens}")
        if not tokenized_corpus:
            print("  警告: 有効なトークン化済みチャンクが一つもありませんでした。")
        print(f"BM25用コーパスのトークン化完了。処理できた有効チャンク数: {len(successfully_processed_chunks)} / {len(self.chunks)}")
        return tokenized_corpus, successfully_processed_chunks

    def _load_or_build_bm25_index(self) -> typing.Union[BM25Okapi, None]: # ★修正
        loaded_from_file = False
        if self.tokenized_corpus_file_path.exists():
            print(f"トークン化済みコーパスをファイルからロード中: {self.tokenized_corpus_file_path}")
            try:
                with open(self.tokenized_corpus_file_path, 'rb') as f:
                    saved_data = pickle.load(f)
                if isinstance(saved_data, dict) and \
                   'tokenized_corpus' in saved_data and \
                   'processed_chunk_ids' in saved_data:
                    self.tokenized_corpus_for_bm25 = saved_data['tokenized_corpus']
                    original_chunks_map = {c['id']: c for c in self.chunks}
                    self.chunks = []
                    for cid in saved_data['processed_chunk_ids']:
                        if cid in original_chunks_map:
                            self.chunks.append(original_chunks_map[cid])
                    print(f"  トークン化済みコーパス ({len(self.tokenized_corpus_for_bm25)}件) と "
                          f"対応チャンク ({len(self.chunks)}件) をロードしました。")
                    if len(self.tokenized_corpus_for_bm25) != len(self.chunks):
                        print("  警告: ロードしたトークン化コーパスとチャンク数が不一致。インデックス再構築を推奨。")
                        self.tokenized_corpus_for_bm25 = None
                    else:
                        loaded_from_file = True
                else:
                    print("  警告: トークン化済みコーパスファイルの形式が不正。再構築します。")
            except Exception as e:
                print(f"  トークン化済みコーパスのロード失敗: {e}. 再構築します。")

        if not loaded_from_file:
            print("トークン化済みコーパスを生成・保存します...")
            self.tokenized_corpus_for_bm25, filtered_chunks = self._create_tokenized_corpus_and_filter_chunks()
            self.chunks = filtered_chunks
            if self.tokenized_corpus_for_bm25 and self.chunks :
                try:
                    data_to_save = {
                        'tokenized_corpus': self.tokenized_corpus_for_bm25,
                        'processed_chunk_ids': [c['id'] for c in self.chunks]
                    }
                    with open(self.tokenized_corpus_file_path, 'wb') as f:
                        pickle.dump(data_to_save, f)
                    print(f"  トークン化済みコーパスを保存しました: {self.tokenized_corpus_file_path}")
                except Exception as e:
                    print(f"  トークン化済みコーパスの保存失敗: {e}")
            elif not self.chunks:
                 print("  警告: トークン化・フィルタリングの結果、有効なチャンクがありません。BM25インデックスは構築できません。")
                 return None
            else:
                 print("  警告: トークン化済みコーパスが空です。BM25インデックスは構築できません。")
                 return None

        if self.bm25_index_file_path.exists():
            print(f"BM25インデックスをファイルからロード中: {self.bm25_index_file_path}")
            try:
                with open(self.bm25_index_file_path, 'rb') as f:
                    bm25_index_loaded = pickle.load(f)
                if isinstance(bm25_index_loaded, BM25Okapi):
                    print("  BM25インデックスのロード完了。")
                    return bm25_index_loaded
                else:
                    print("  警告: ロードしたBM25インデックスの型が不正。再構築します。")
            except Exception as e:
                print(f"  BM25インデックスのロードに失敗: {e}. 再構築します。")
        
        if not self.tokenized_corpus_for_bm25 or not self.chunks:
            print("  警告: BM25インデックス構築に必要なトークン化済みコーパスまたはチャンクデータがありません。")
            return None
            
        print("BM25インデックスを新規構築中...")
        try:
            bm25_index_new = BM25Okapi(self.tokenized_corpus_for_bm25)
            print("  BM25インデックス構築完了。")
            if self.tokenized_corpus_for_bm25:
                 sample_query_toks = tokenize_text_for_bm25_internal("テスト")
                 if not (len(sample_query_toks) == 1 and sample_query_toks[0].startswith("<bm25_")):
                    test_scrs = bm25_index_new.get_scores(sample_query_toks)
                    print(f"    構築直後のテスト検索スコア (上位3件, クエリ: '{sample_query_toks}'): {test_scrs[:3] if test_scrs is not None else 'N/A'}")
            try:
                with open(self.bm25_index_file_path, 'wb') as f:
                    pickle.dump(bm25_index_new, f)
                print(f"  BM25インデックスをファイルに保存しました: {self.bm25_index_file_path}")
            except Exception as e:
                print(f"  BM25インデックスの保存に失敗: {e}")
            return bm25_index_new
        except ZeroDivisionError:
            print("  警告: BM25インデックス構築中にZeroDivisionError。BM25は機能しない可能性があります。")
            return None
        except Exception as e:
            print(f"  BM25インデックス構築中に予期せぬエラー: {e}")
            traceback.print_exc()
            return None
            
    def get_embedding_from_openai(self, text: str, model_name: typing.Union[str, None] = None, client = None) -> typing.Union[list[float], None]: # ★修正
        if model_name is None: model_name = self.embedding_model
        if client is None:
            try:
                from openai import OpenAI
                api_key_env = os.getenv("OPENAI_API_KEY")
                if not api_key_env:
                    print("  警告 (get_embedding): OPENAI_API_KEY が未設定。埋め込み取得不可。")
                    return None
                client = OpenAI(api_key=api_key_env)
            except Exception as e_client_init:
                print(f"  OpenAI Client初期化エラー (get_embedding内): {e_client_init}")
                return None
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            print(f"  警告 (get_embedding): 埋め込み対象のテキストが空または不正。")
            return None
        try:
            response = client.embeddings.create(model=model_name, input=text) # type: ignore
            embedding = response.data[0].embedding
            return embedding
        except Exception as e_openai_emb:
            print(f"  OpenAI API埋め込みエラー: model={model_name}, text(先頭30字)='{text[:30]}...' Error: {e_openai_emb}")
            return None

    def search(self, query: str, top_k: int = 5, threshold: float = 0.15, 
               vector_weight: float = 0.7, bm25_weight: float = 0.3, 
               client = None) -> tuple[list[dict], bool]:
        print(f"検索実行: クエリ='{query}', top_k={top_k}, threshold={threshold}, vec_w={vector_weight}, bm25_w={bm25_weight}")
        if not self.chunks:
            print("  警告: 有効なチャンクデータが存在しません (BM25処理後)。検索を中止します。")
            return [], True
        
        query_vector = self.get_embedding_from_openai(query, client=client)
        if query_vector is None:
            if self.model:
                print("  OpenAI APIでのベクトル化失敗。バックアップモデル (SentenceTransformer) を使用します。")
                try:
                    query_vector = self.model.encode(query).tolist()
                    print(f"    バックアップベクトル化成功: dim={len(query_vector)}")
                except Exception as e_st_encode:
                    print(f"    バックアップベクトル化エラー: {e_st_encode}")
                    return [], True
            else:
                print("  クエリをベクトル化できませんでした (OpenAI失敗、バックアップモデルなし)。")
                return [], True
        
        vector_scores: dict[str, float] = {}
        for chunk_data in self.chunks:
            chunk_id = chunk_data['id']
            if chunk_id in self.embeddings:
                chunk_vector = self.embeddings[chunk_id]
                try:
                    q_vec_arr = np.array(query_vector, dtype=np.float32).flatten()
                    c_vec_arr = np.array(chunk_vector, dtype=np.float32).flatten()
                    if q_vec_arr.shape[0] != c_vec_arr.shape[0]:
                        similarity = 0.0
                    elif q_vec_arr.shape[0] == 0:
                        similarity = 0.0
                    else:
                        dot_product = np.dot(q_vec_arr, c_vec_arr)
                        norm_q = np.linalg.norm(q_vec_arr)
                        norm_c = np.linalg.norm(c_vec_arr)
                        if norm_q > 1e-9 and norm_c > 1e-9:
                            similarity = dot_product / (norm_q * norm_c)
                        else:
                            similarity = 0.0
                    vector_scores[chunk_id] = float(similarity)
                except Exception as e_cosine: print(f"    コサイン類似度計算エラー (ID:{chunk_id}): {e_cosine}"); vector_scores[chunk_id] = 0.0

        bm25_scores_map: dict[str, float] = {}
        if self.bm25_index and self.tokenized_corpus_for_bm25 and self.chunks:
            query_tokens_for_bm25 = tokenize_text_for_bm25_internal(query)
            print(f"  BM25用クエリトークン (SudachiPy使用): {query_tokens_for_bm25}")
            is_dummy_query_token = len(query_tokens_for_bm25) == 1 and \
                                   query_tokens_for_bm25[0].startswith("<bm25_") and \
                                   query_tokens_for_bm25[0].endswith("_token>")
            if query_tokens_for_bm25 and not is_dummy_query_token:
                try:
                    raw_bm25_scores_from_lib = self.bm25_index.get_scores(query_tokens_for_bm25)
                    if raw_bm25_scores_from_lib is not None and len(raw_bm25_scores_from_lib) == len(self.chunks):
                        for i, chunk_data_bm25 in enumerate(self.chunks):
                            bm25_scores_map[chunk_data_bm25['id']] = float(raw_bm25_scores_from_lib[i])
                        all_bm25_vals = np.array(list(bm25_scores_map.values()), dtype=np.float32)
                        if all_bm25_vals.size > 0 :
                            max_bm25_score = np.max(all_bm25_vals)
                            if max_bm25_score > 1e-9:
                                for cid in bm25_scores_map:
                                    bm25_scores_map[cid] /= max_bm25_score
                    elif raw_bm25_scores_from_lib is None:
                         print("    警告: BM25ライブラリ (get_scores) が None を返しました。")
                    else:
                        print(f"    致命的エラー: BM25スコアリスト長 ({len(raw_bm25_scores_from_lib)}) と "
                              f"有効チャンク数 ({len(self.chunks)}) が不一致。BM25スコアは使用できません。")
                except Exception as e_bm25_search:
                    print(f"    BM25検索処理中に予期せぬエラー: {e_bm25_search}"); traceback.print_exc()
            else:
                print("    BM25検索用の有効なクエリトークンがないため、BM25スコアは0として扱います。")
        else:
            print("    BM25インデックスまたはトークン化済みコーパスが見つからないため、BM25検索をスキップします。")

        hybrid_scores_data: list[dict] = []
        for chunk_data_final in self.chunks:
            chunk_id_final = chunk_data_final['id']
            vec_s = vector_scores.get(chunk_id_final, 0.0)
            bm25_s_final = bm25_scores_map.get(chunk_id_final, 0.0)
            current_hybrid_score = (vector_weight * vec_s) + (bm25_weight * bm25_s_final)
            hybrid_scores_data.append({
                'chunk': chunk_data_final,
                'similarity': current_hybrid_score,
                'vector_score': vec_s,
                'bm25_score': bm25_s_final
            })
        hybrid_scores_data.sort(key=lambda x: x['similarity'], reverse=True)
        print(f"  上位ハイブリッドスコア (ソート後):")
        for i, score_item in enumerate(hybrid_scores_data[:min(5, len(hybrid_scores_data))]):
           print(f"    {i+1}. ID: {score_item['chunk']['id']}, "
                 f"Hybrid: {score_item['similarity']:.4f} "
                 f"(Vec: {score_item['vector_score']:.4f}, BM25: {score_item['bm25_score']:.4f})")
        
        final_filtered_results = [
            r_item for r_item in hybrid_scores_data if r_item['similarity'] >= threshold
        ][:top_k]
        print(f"  閾値({threshold})以上かつTopK({top_k})の結果件数: {len(final_filtered_results)}")
        
        output_results: list[dict] = []
        for r_final_item in final_filtered_results:
            output_results.append({
                'id': r_final_item['chunk']['id'],
                'text': r_final_item['chunk']['text'],
                'metadata': r_final_item['chunk']['metadata'],
                'similarity': r_final_item['similarity'],
                'vector_score': r_final_item['vector_score'],
                'bm25_score': r_final_item['bm25_score']
            })
        is_not_found = len(output_results) == 0
        if is_not_found and hybrid_scores_data and top_k > 0:
            print("    閾値を超える結果がないため、最も類似度の高い結果を1件返します (閾値未満の可能性あり)。")
            best_match_item = hybrid_scores_data[0]
            output_results = [{
                'id': best_match_item['chunk']['id'],
                'text': best_match_item['chunk']['text'],
                'metadata': best_match_item['chunk']['metadata'],
                'similarity': best_match_item['similarity'],
                'vector_score': best_match_item['vector_score'],
                'bm25_score': best_match_item['bm25_score']
            }]
            is_not_found = False
        return output_results, is_not_found

def search_knowledge_base(query: str, kb_path: str, top_k: int = 5, threshold: float = 0.15, 
                          embedding_model: typing.Union[str, None] = None, client = None) -> tuple[list[dict], bool]: # ★修正
    try:
        print("\n" + "="*50)
        print(f"ナレッジベース検索開始: クエリ='{query}'")
        print(f"  KBパス='{kb_path}', TopK={top_k}, 閾値={threshold}, EmbModel={embedding_model}")
        resolved_kb_path = Path(kb_path).resolve()
        if not resolved_kb_path.exists():
            print(f"  エラー: 指定されたナレッジベースパスが見つかりません: {resolved_kb_path}")
            return [], True
        print("  検索エンジンを初期化中...")
        search_engine = HybridSearchEngine(str(resolved_kb_path)) 
        print("  検索を実行中...")
        results, not_found = search_engine.search(query, top_k, threshold, client=client)
        print(f"検索完了: 結果{len(results)}件, 見つからなかったフラグ: {not_found}")
        print("="*50 + "\n")
        return results, not_found
    except Exception as e_skb:
        print(f"検索処理全体でエラー (search_knowledge_base): {type(e_skb).__name__}: {e_skb}")
        print("スタックトレース:"); traceback.print_exc()
        return [], True

def get_openai_client_for_kb_search():
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("警告 (get_openai_client_for_kb_search): OPENAI_API_KEYが環境変数に設定されていません。")
            return None
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"OpenAIクライアント初期化エラー (get_openai_client_for_kb_search): {e}")
        return None

if __name__ == "__main__":
    print("knowledge_search.py を直接実行します (テストモード)")
    script_dir = Path(__file__).resolve().parent
    default_test_kb_relative_path = "../knowledge_base/default_kb"
    test_kb_full_path = (script_dir / default_test_kb_relative_path).resolve()
    print(f"テスト用ナレッジベースのパス: {test_kb_full_path}")
    if not test_kb_full_path.exists() or not test_kb_full_path.is_dir():
        print(f"エラー: テスト用ナレッジベースパス {test_kb_full_path} が見つからないか、ディレクトリではありません。")
    else:
        test_query_example = "特定の技術に関する情報はありますか"
        print(f"テスト検索を実行します。クエリ: '{test_query_example}'")
        results_list, not_found_flag_result = search_knowledge_base(
            test_query_example, 
            str(test_kb_full_path),
            client=None
        )
        if not_found_flag_result:
            print("テスト結果: 検索結果は見つかりませんでした。")
        else:
            print(f"テスト結果: {len(results_list)}件の結果が見つかりました。")
            for i, result_item in enumerate(results_list):
                print(f"  結果 {i+1}: ID='{result_item.get('id', 'N/A')}', "
                      f"HybridScore={result_item.get('similarity', 0.0):.4f}, "
                      f"VecScore={result_item.get('vector_score',0.0):.4f}, "
                      f"BM25Score={result_item.get('bm25_score',0.0):.4f}")
                print(f"    Text: {result_item.get('text', '')[:80]}...")