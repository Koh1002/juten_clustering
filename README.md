# テキストクラスタリング アプリケーション

Talk to the City風のワークフローを参考にした、Excelテキストデータのクラスタリング・可視化アプリケーションです。

## 機能

- **Excel読み込み**: 結合セルを自動解除し、「提出用」シートからデータを抽出
- **カテゴリ列自動探索**: R列周辺からA〜Eカテゴリを自動検出
- **クラスタリング**: Embedding → UMAP → HDBSCAN（または TF-IDF + KMeans）
- **ラベル生成**: LLMによるクラスタラベル・説明文の自動生成
- **可視化**: UMAP 2D散布図、カテゴリ分布、クラスタ構成比
- **フィルタ**: 店舗・カテゴリ・ブロック番号による絞り込み
- **レポート**: Talk to the City風のクラスタ一覧ページ

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. アプリケーションの起動

```bash
streamlit run app.py
```

ブラウザで http://localhost:8501 が開きます。

## 使い方

### 1. データ取込

1. 「データ取込/成形」タブを開く
2. Excelファイルをアップロード、または「ファイル/フォルダを選択」ボタンで選択
3. 処理完了後、成形済みデータがプレビューされます

### 2. APIキー設定（オプション）

サイドバーでAPIキーを入力すると、高品質な埋め込みとラベル生成が可能になります。

- **OpenAI**: `sk-xxx` 形式
- **Gemini**: `AIza...` 形式
- **Claude**: `sk-ant-xxx` 形式（埋め込みは非対応）

APIキーなしでも TF-IDF + KMeans モードで動作します。

### 3. クラスタリング実行

1. 「可視化」タブを開く
2. サイドバーでフィルタを設定（任意）
3. 「クラスタリング実行」ボタンをクリック
4. UMAP散布図とクラスタ情報が表示されます

### 4. レポート確認

「クラスタ一覧レポート」タブで、各クラスタの詳細を確認できます。

- クラスタラベル
- 特徴説明
- 頻出語
- 代表レコード
- 「別アイデア！」ボタンでラベル再生成

## ファイル構成

```
.
├── app.py                    # Streamlitメインアプリ
├── requirements.txt          # 依存パッケージ
├── README.md
├── data/                     # サンプルデータ格納
├── scripts/
│   └── create_sample_excel.py  # サンプルExcel生成スクリプト
└── src/
    ├── __init__.py
    ├── utils.py              # 共通ユーティリティ
    ├── excel_parser.py       # Excel解析（結合セル処理含む）
    ├── cluster.py            # クラスタリングロジック
    ├── labeler.py            # ラベル生成・頻出語抽出
    ├── viz.py                # 可視化（Plotly）
    └── llm/
        ├── __init__.py
        └── providers.py      # LLMプロバイダ抽象化
```

## サンプルExcelの作成

```bash
# 単一ファイル
python scripts/create_sample_excel.py

# 複数ファイル
python scripts/create_sample_excel.py --multiple
```

## Excelフォーマット

読み込み対象のExcelは以下の形式を想定しています：

- シート名: 「提出用」（「作成例」は無視）
- D4セル: 店舗名（結合セル可）
- N4セル: ブロック番号（結合セル可）
- 7行目以降:
  - E列に `1.` `2.` 等の番号 → 概要行（本文はF〜O列）
  - F列に `(1)` `(2)` 等の番号 → 詳細行（本文はG〜O列）
  - R列（またはP〜V列のいずれか）にカテゴリ `【A】` 〜 `【E】`

## カテゴリマッピング

| 記号 | カテゴリ名 |
|------|-----------|
| A | 使命 |
| B | 利益 |
| C | 能力開発 |
| D | 実現力 |
| E | コストコントロール |

## 技術スタック

- **UI**: Streamlit
- **Excel処理**: openpyxl, pandas
- **クラスタリング**: scikit-learn, hdbscan, umap-learn
- **可視化**: Plotly
- **LLM**: OpenAI SDK, google-generativeai, anthropic
- **形態素解析**: Janome

## ライセンス

MIT License
