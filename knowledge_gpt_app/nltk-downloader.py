import nltk

# ナレッジ検索に必要なNLTKリソースをダウンロード
print("NLTKリソースをダウンロードしています...")

# 必要なリソースを全てダウンロード
resources = [
    'punkt',              # テキストトークナイザ
    'averaged_perceptron_tagger',  # 品詞タグ付け
    'stopwords',          # ストップワード
    'wordnet',            # WordNet辞書
    'omw-1.4'             # Open Multilingual WordNet
]

for resource in resources:
    print(f"{resource}をダウンロード中...")
    nltk.download(resource)
    print(f"{resource}のダウンロードが完了しました。")

print("全てのリソースのダウンロードが完了しました。これでナレッジ検索が正常に動作するはずです。")