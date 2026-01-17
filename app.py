"""
テキストクラスタリング Streamlit アプリケーション

Talk to the City風のワークフロー:
Embedding → 次元圧縮 → クラスタリング → ラベリング → 可視化
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import sys
import os

# srcをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.excel_parser import ExcelParser, select_files_with_dialog
from src.cluster import TextClusterer, ClusteringResult
from src.labeler import ClusterLabeler
from src.llm.providers import detect_provider, LLMProvider
from src.viz import (
    create_scatter_plot,
    create_category_distribution_chart,
    create_cluster_size_pie,
    export_to_csv
)
from src.utils import create_document_text, truncate_text, logger


# ページ設定
st.set_page_config(
    page_title="Text Clustering",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS - シャープでモダンなデザイン
st.markdown("""
<style>
/* 全体のフォント・背景 */
html, body, [class*="css"] {
    font-family: 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', sans-serif;
}

/* メインコンテンツ - 上部余白を最小化 */
.main .block-container {
    padding-top: 0.5rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

/* Streamlitのデフォルトヘッダー余白を削除 */
.stApp > header {
    height: 0 !important;
}

.main > div:first-child {
    padding-top: 0 !important;
}

/* ヘッダー・タイトル */
h1 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
    margin-top: 0 !important;
    margin-bottom: 0.3rem !important;
    letter-spacing: -0.02em;
}

h2 {
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    color: #333 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: #444 !important;
}

/* サイドバー - 深い緑 */
[data-testid="stSidebar"] {
    background-color: #1a3a2f !important;
}

[data-testid="stSidebar"] * {
    color: #e8f0ed !important;
}

[data-testid="stSidebar"] .stTextInput > div > div > input {
    background-color: #2d4f42 !important;
    border: 1px solid #3d6354 !important;
    color: #fff !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #2d4f42 !important;
    border: 1px solid #3d6354 !important;
}

[data-testid="stSidebar"] .stMultiSelect > div > div {
    background-color: #2d4f42 !important;
    border: 1px solid #3d6354 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #3d6354 !important;
}

[data-testid="stSidebar"] .stCheckbox label {
    color: #c8d8d2 !important;
}

/* ボタン */
.stButton > button {
    background-color: #1a3a2f !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.4rem 1rem !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    transition: background-color 0.2s;
}

.stButton > button:hover {
    background-color: #2d5244 !important;
}

.stButton > button[kind="primary"] {
    background-color: #1a3a2f !important;
}

/* タブ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #ddd;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1.2rem;
    font-size: 0.85rem;
    font-weight: 400;
    color: #666;
    border-bottom: 2px solid transparent;
    background-color: transparent;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #1a3a2f;
}

.stTabs [aria-selected="true"] {
    color: #1a3a2f !important;
    border-bottom-color: #1a3a2f !important;
    background-color: transparent !important;
}

/* メトリクス */
[data-testid="stMetric"] {
    background-color: #f8faf9;
    padding: 0.8rem;
    border-radius: 4px;
    border-left: 3px solid #1a3a2f;
}

[data-testid="stMetric"] label {
    font-size: 0.75rem !important;
    color: #666 !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 500 !important;
    color: #1a3a2f !important;
}

/* データフレーム */
.stDataFrame {
    font-size: 0.8rem;
}

/* エキスパンダー */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    color: #444 !important;
}

/* アラート */
.stAlert {
    border-radius: 4px;
    font-size: 0.85rem;
}

/* クラスタカード */
.cluster-card {
    background-color: #fafbfa;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    border-left: 3px solid #1a3a2f;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.cluster-header {
    font-size: 1rem;
    font-weight: 500;
    color: #1a3a2f;
    margin-bottom: 0.5rem;
}

.word-chip {
    display: inline-block;
    background-color: #e8f0ed;
    color: #1a3a2f;
    padding: 3px 10px;
    margin: 3px;
    border-radius: 3px;
    font-size: 0.8rem;
}

.stats-box {
    background-color: #f8faf9;
    border-radius: 4px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #e8f0ed;
}

/* ダウンロードボタン */
.stDownloadButton > button {
    background-color: transparent !important;
    color: #1a3a2f !important;
    border: 1px solid #1a3a2f !important;
}

.stDownloadButton > button:hover {
    background-color: #f0f5f3 !important;
}

/* スピナー */
.stSpinner > div {
    border-top-color: #1a3a2f !important;
}

/* 成功・警告・エラーメッセージ */
.stSuccess {
    background-color: #e8f5e9 !important;
    color: #1b5e20 !important;
}

.stWarning {
    background-color: #fff8e1 !important;
    color: #f57f17 !important;
}

.stError {
    background-color: #ffebee !important;
    color: #c62828 !important;
}

/* ファイルアップローダー非表示（使わないため） */
[data-testid="stFileUploader"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """セッション状態を初期化"""
    defaults = {
        'df': None,
        'df_filtered': None,
        'clustering_result': None,
        'cluster_info': None,
        'llm_provider': None,
        'provider_status': '',
        'parse_errors': [],
        'parse_warnings': [],
        'category_column_info': {},
        'use_api_mode': True,
        'recalculate_on_filter': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """サイドバーを描画"""
    st.sidebar.title("設定")

    # APIキー入力
    st.sidebar.subheader("APIキー")
    api_key = st.sidebar.text_input(
        "APIキー（OpenAI/Gemini/Claude）",
        type="password",
        help="OpenAI、Gemini、Claudeいずれかのキーを入力"
    )

    if api_key:
        if st.sidebar.button("キーを検証"):
            with st.spinner("検証中..."):
                provider, status = detect_provider(api_key)
                st.session_state.llm_provider = provider
                st.session_state.provider_status = status

                if provider:
                    # モデル情報を取得
                    provider.list_models()
                    provider.pick_latest('embedding')
                    provider.pick_latest('generation')

    # プロバイダ情報表示
    if st.session_state.provider_status:
        if st.session_state.llm_provider:
            st.sidebar.success(st.session_state.provider_status)
            info = st.session_state.llm_provider.get_model_info()
            st.sidebar.info(f"""
            **プロバイダ**: {info['provider']}
            **埋め込みモデル**: {info['embedding_model'] or 'N/A'}
            **生成モデル**: {info['generation_model'] or 'N/A'}
            **更新時刻**: {info['last_updated'] or 'N/A'}
            """)

            if info['top_models']:
                with st.sidebar.expander("モデル候補（上位5件）"):
                    for m in info['top_models'][:5]:
                        st.write(f"- {m.get('id', m)}")
        else:
            st.sidebar.error(st.session_state.provider_status)

    st.sidebar.divider()

    # クラスタリング設定
    st.sidebar.subheader("クラスタリング設定")

    use_api = st.sidebar.checkbox(
        "API埋め込みを使用",
        value=st.session_state.use_api_mode,
        help="OFFにするとTF-IDF+KMeansを使用（APIキー不要）"
    )
    st.session_state.use_api_mode = use_api

    use_hdbscan = st.sidebar.checkbox(
        "HDBSCANを使用",
        value=True,
        help="OFFにするとKMeansを使用"
    )

    # HDBSCANパラメータ
    with st.sidebar.expander("詳細パラメータ"):
        min_cluster_size = st.slider("HDBSCAN min_cluster_size", 2, 20, 5)
        min_samples = st.slider("HDBSCAN min_samples", 1, 10, 3)
        kmeans_n = st.slider("KMeans クラスタ数", 2, 20, 8)

    st.sidebar.divider()

    # フィルタ設定
    st.sidebar.subheader("フィルタ")

    filters = {
        'stores': [],
        'categories': [],
        'blocks': [],
        'doc_type': 'both'
    }

    if st.session_state.df is not None:
        df = st.session_state.df

        # 店舗フィルタ
        if '店舗名' in df.columns:
            stores = ['すべて'] + sorted(df['店舗名'].unique().tolist())
            selected_stores = st.sidebar.multiselect(
                "店舗",
                stores,
                default=['すべて']
            )
            if 'すべて' not in selected_stores:
                filters['stores'] = selected_stores

        # カテゴリフィルタ
        if 'カテゴリ名' in df.columns:
            categories = ['すべて'] + sorted(df['カテゴリ名'].unique().tolist())
            selected_cats = st.sidebar.multiselect(
                "カテゴリ",
                categories,
                default=['すべて']
            )
            if 'すべて' not in selected_cats:
                filters['categories'] = selected_cats

        # ブロック番号フィルタ
        if 'ブロック番号' in df.columns:
            blocks = ['すべて'] + sorted(df['ブロック番号'].unique().tolist())
            selected_blocks = st.sidebar.multiselect(
                "ブロック番号",
                blocks,
                default=['すべて']
            )
            if 'すべて' not in selected_blocks:
                filters['blocks'] = selected_blocks

        # 文書タイプ
        doc_type = st.sidebar.radio(
            "文書タイプ",
            ['両方', '概要のみ', '詳細のみ'],
            horizontal=True
        )
        filters['doc_type'] = doc_type

    st.sidebar.divider()

    # フィルタ時の再計算
    recalc = st.sidebar.checkbox(
        "フィルタ変更時にクラスタを再計算",
        value=st.session_state.recalculate_on_filter,
        help="ONにするとフィルタ後のデータで再クラスタリング"
    )
    st.session_state.recalculate_on_filter = recalc

    return {
        'use_api': use_api,
        'use_hdbscan': use_hdbscan,
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'kmeans_n': kmeans_n,
        'filters': filters
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """フィルタを適用"""
    filtered = df.copy()

    if filters['stores']:
        filtered = filtered[filtered['店舗名'].isin(filters['stores'])]

    if filters['categories']:
        filtered = filtered[filtered['カテゴリ名'].isin(filters['categories'])]

    if filters['blocks']:
        filtered = filtered[filtered['ブロック番号'].isin(filters['blocks'])]

    if filters['doc_type'] == '概要のみ':
        filtered = filtered[filtered['取り組み宣言（詳細）'] == '']
    elif filters['doc_type'] == '詳細のみ':
        filtered = filtered[filtered['取り組み宣言（詳細）'] != '']

    return filtered


def render_tab_import():
    """データ取込/成形タブ"""
    st.header("データ取込・成形")

    st.subheader("Excelファイル選択")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("ファイル/フォルダを選択", type="primary", key="btn_dialog"):
            try:
                file_paths, mode = select_files_with_dialog()
                if file_paths:
                    st.session_state['selected_files'] = file_paths
                    st.success(f"{len(file_paths)}件のファイルを選択（{mode}）")
                else:
                    st.warning("ファイルが選択されませんでした")
            except Exception as e:
                st.error(f"ダイアログエラー: {e}")

    with col2:
        if 'selected_files' in st.session_state and st.session_state['selected_files']:
            if st.button("選択をクリア", key="btn_clear"):
                st.session_state['selected_files'] = []
                st.rerun()

    # 選択ファイル表示と処理
    if 'selected_files' in st.session_state and st.session_state['selected_files']:
        with st.expander(f"選択済みファイル（{len(st.session_state['selected_files'])}件）", expanded=True):
            for f in st.session_state['selected_files']:
                st.write(f"- {os.path.basename(f)}")

        if st.button("ファイルを処理", type="primary", key="btn_process_local"):
            process_files(file_paths=st.session_state['selected_files'])

    # 処理結果表示
    if st.session_state.parse_errors:
        st.error("エラー:")
        for err in st.session_state.parse_errors:
            st.write(f"- {err}")

    if st.session_state.parse_warnings:
        st.warning("警告:")
        for warn in st.session_state.parse_warnings:
            st.write(f"- {warn}")

    # カテゴリ列情報
    if st.session_state.category_column_info:
        with st.expander("カテゴリ列探索結果"):
            info = st.session_state.category_column_info
            if 'best' in info:
                best = info['best']
                st.write(f"**採用列**: {best.get('column', 'N/A')}")
                st.write(f"**マッチ率**: {best.get('match_rate', 0):.1%}")
                st.write(f"**検出例**: {', '.join(best.get('examples', []))}")

            if 'candidates' in info:
                st.write("**候補列スコア:**")
                for cand in info['candidates'][:5]:
                    st.write(f"- {cand['column']}: {cand['score']:.3f}")

    # データプレビュー
    if st.session_state.df is not None:
        st.subheader("成形済みデータプレビュー")
        df = st.session_state.df

        st.write(f"**総レコード数**: {len(df)}")

        # 統計情報
        col1, col2, col3 = st.columns(3)
        with col1:
            if '店舗名' in df.columns:
                st.metric("店舗数", df['店舗名'].nunique())
        with col2:
            if 'カテゴリ名' in df.columns:
                st.metric("カテゴリ数", df['カテゴリ名'].nunique())
        with col3:
            summary_only = (df['取り組み宣言（詳細）'] == '').sum()
            st.metric("概要のみレコード", summary_only)

        # データテーブル
        display_cols = ['店舗名', 'ブロック番号', 'カテゴリ', 'カテゴリ名',
                       '取り組み宣言（概要）', '取り組み宣言（詳細）']
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols].head(50), use_container_width=True)

        # CSVダウンロード
        csv_data = export_to_csv(df, include_cluster=False)
        st.download_button(
            "成形済みCSVをダウンロード",
            csv_data,
            file_name=f"formatted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def process_files(file_paths=None, file_objects=None):
    """ファイルを処理してDataFrameを作成"""
    with st.spinner("Excelファイルを処理中..."):
        try:
            parser = ExcelParser()
            df = parser.load_files(file_paths=file_paths, file_objects=file_objects)

            # 文書テキスト列を追加（クラスタリング用）
            df['_document_text'] = df.apply(
                lambda row: create_document_text(
                    row.get('カテゴリ名', ''),
                    row.get('取り組み宣言（概要）', ''),
                    row.get('取り組み宣言（詳細）', '')
                ),
                axis=1
            )

            st.session_state.df = df
            st.session_state.parse_errors = parser.get_errors()
            st.session_state.parse_warnings = parser.get_warnings()
            st.session_state.category_column_info = parser.get_category_column_info()

            st.success(f"処理完了: {len(df)}件のレコードを抽出")
            st.rerun()

        except Exception as e:
            st.error(f"処理エラー: {e}")
            logger.exception("File processing error")


def render_tab_visualization(settings: dict):
    """可視化タブ"""
    st.header("可視化")

    if st.session_state.df is None:
        st.info("まず「データ取込」タブでファイルを読み込んでください")
        return

    df = st.session_state.df
    filters = settings['filters']

    # フィルタ適用
    df_filtered = apply_filters(df, filters)
    st.session_state.df_filtered = df_filtered

    st.write(f"**表示レコード数**: {len(df_filtered)} / {len(df)}")

    if len(df_filtered) == 0:
        st.warning("フィルタ条件に一致するデータがありません")
        return

    # クラスタリング設定と実行
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        cluster_method = st.radio(
            "手法",
            ["KMeans（数指定）", "HDBSCAN（自動）"],
            index=0,
            horizontal=True
        )
        use_hdbscan = cluster_method == "HDBSCAN（自動）"

    with col2:
        n_clusters = st.number_input(
            "クラスタ数",
            min_value=2,
            max_value=30,
            value=8,
            step=1,
            disabled=use_hdbscan,
            help="KMeans使用時のクラスタ数"
        )
        settings['kmeans_n'] = int(n_clusters)
        settings['use_hdbscan'] = use_hdbscan

    with col3:
        run_clustering = st.button("クラスタリング実行", type="primary")

    with col4:
        if settings['use_api'] and not st.session_state.llm_provider:
            st.warning("APIキー未設定")
        elif settings['use_api'] and st.session_state.llm_provider and not st.session_state.llm_provider.supports_embedding:
            st.warning("埋め込み非対応")

    if run_clustering:
        run_clustering_pipeline(df_filtered, settings)

    # 結果表示
    if st.session_state.clustering_result is not None:
        result = st.session_state.clustering_result
        cluster_info = st.session_state.cluster_info

        # サマリ統計
        st.subheader("クラスタリング結果")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("クラスタ数", result.n_clusters)
        with col2:
            st.metric("ノイズ数", result.n_noise)
        with col3:
            st.metric("手法", result.method.upper())
        with col4:
            if result.silhouette:
                st.metric("シルエットスコア", f"{result.silhouette:.3f}")

        # 散布図
        st.subheader("UMAP 2D散布図")

        # 散布図オプション
        opt_col1, opt_col2 = st.columns([2, 1])

        with opt_col1:
            # ハイライトクラスタ選択
            cluster_options = ['すべて表示'] + [
                f"{cid}: {cluster_info.get(cid, {}).get('label', f'クラスタ{cid}')}"
                for cid in sorted(set(result.labels)) if cid != -1
            ]
            highlight_selection = st.selectbox("クラスタをハイライト", cluster_options)

        with opt_col2:
            # ラベル表示切り替え
            show_labels = st.checkbox("クラスタ名を表示", value=True)

        highlight_cluster = None
        if highlight_selection != 'すべて表示':
            highlight_cluster = int(highlight_selection.split(':')[0])

        fig = create_scatter_plot(
            df_filtered,
            result.embeddings_2d,
            result.labels,
            cluster_info,
            highlight_cluster=highlight_cluster,
            show_labels=show_labels
        )
        st.plotly_chart(fig, use_container_width=True)

        # 追加チャート
        col1, col2 = st.columns(2)

        with col1:
            fig_pie = create_cluster_size_pie(result.labels, cluster_info)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_cat = create_category_distribution_chart(df_filtered, result.labels, cluster_info)
            st.plotly_chart(fig_cat, use_container_width=True)

        # CSVダウンロード
        st.subheader("ダウンロード")

        col1, col2 = st.columns(2)

        with col1:
            csv_with_cluster = export_to_csv(
                df_filtered, result.labels, cluster_info, include_cluster=True
            )
            st.download_button(
                "クラスタ付きCSV（フィルタ後）",
                csv_with_cluster,
                file_name=f"clustered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            # 全件版
            if len(df_filtered) != len(df):
                st.info("全件データにはクラスタ列が含まれません")


def run_clustering_pipeline(df: pd.DataFrame, settings: dict):
    """クラスタリングパイプラインを実行"""
    with st.spinner("クラスタリング中..."):
        try:
            # テキスト取得
            texts = df['_document_text'].tolist()

            if len(texts) < 3:
                st.error("クラスタリングには最低3件のレコードが必要です")
                return

            # クラスタラー初期化
            use_api = settings['use_api'] and st.session_state.llm_provider is not None
            if use_api and st.session_state.llm_provider:
                if not st.session_state.llm_provider.supports_embedding:
                    st.warning("選択されたプロバイダは埋め込みをサポートしていません。TF-IDFを使用します。")
                    use_api = False

            clusterer = TextClusterer(
                use_api=use_api,
                llm_provider=st.session_state.llm_provider if use_api else None,
                hdbscan_min_cluster_size=settings['min_cluster_size'],
                hdbscan_min_samples=settings['min_samples'],
                kmeans_n_clusters=settings['kmeans_n']
            )

            # クラスタリング実行
            result = clusterer.fit(texts, use_hdbscan=settings['use_hdbscan'])
            st.session_state.clustering_result = result

            # ラベル生成
            st.info("クラスタラベルを生成中...")
            labeler = ClusterLabeler(llm_provider=st.session_state.llm_provider)
            cluster_info = labeler.generate_all_labels(df, result.labels)
            st.session_state.cluster_info = cluster_info

            st.success("クラスタリング完了！")
            st.rerun()

        except Exception as e:
            st.error(f"クラスタリングエラー: {e}")
            logger.exception("Clustering error")


def render_tab_report():
    """クラスタ一覧レポートタブ"""
    st.header("クラスタ一覧レポート")

    if st.session_state.clustering_result is None:
        st.info("まず「可視化」タブでクラスタリングを実行してください")
        return

    result = st.session_state.clustering_result
    cluster_info = st.session_state.cluster_info
    df_filtered = st.session_state.df_filtered

    if df_filtered is None:
        df_filtered = st.session_state.df

    # 全体サマリ
    st.subheader("全体サマリ")

    labeler = ClusterLabeler()
    summary = labeler.get_summary_stats(df_filtered, result.labels)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総レコード数", summary['total_records'])
    with col2:
        st.metric("クラスタ数", summary['n_clusters'])
    with col3:
        st.metric("ノイズ数", summary['n_noise'])
    with col4:
        st.metric("ノイズ率", f"{summary['noise_ratio']:.1%}")

    # カテゴリ別件数
    if summary['category_counts']:
        with st.expander("カテゴリ別件数"):
            for cat, count in summary['category_counts'].items():
                st.write(f"- {cat}: {count}件")

    st.divider()

    # クラスタカード
    st.subheader("クラスタ詳細")

    # 別アイデアボタン用の状態
    if 'regenerate_cluster' not in st.session_state:
        st.session_state.regenerate_cluster = None

    for cluster_id in sorted(cluster_info.keys()):
        info = cluster_info[cluster_id]

        # カード風表示
        with st.container():
            st.markdown(f"""
            <div class="cluster-card">
                <div class="cluster-header">
                    {info['label']} (ID: {cluster_id})
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**件数**: {info['count']}件")
                st.write(f"**特徴説明**:")
                st.write(info.get('description', '説明なし'))

            with col2:
                # 別アイデアボタン
                if cluster_id != -1 and st.session_state.llm_provider:
                    if st.button("別アイデア", key=f"regen_{cluster_id}"):
                        regenerate_label(cluster_id, df_filtered, result.labels)

                # ハイライトボタン
                if st.button("散布図で表示", key=f"highlight_{cluster_id}"):
                    st.session_state['highlight_cluster'] = cluster_id
                    st.info(f"「可視化」タブでクラスタ {cluster_id} をハイライトします")

            # 頻出語
            if info.get('frequent_words'):
                st.write("**頻出語**:")
                words_html = " ".join([
                    f'<span class="word-chip">{word}</span>'
                    for word in info['frequent_words']
                ])
                st.markdown(words_html, unsafe_allow_html=True)

            # 代表レコード
            if info.get('representative_texts'):
                with st.expander("代表レコード"):
                    for i, text in enumerate(info['representative_texts'], 1):
                        st.write(f"{i}. {truncate_text(text, 200)}")

            st.divider()


def regenerate_label(cluster_id: int, df: pd.DataFrame, labels: np.ndarray):
    """ラベルを再生成"""
    with st.spinner(f"クラスタ {cluster_id} のラベルを再生成中..."):
        try:
            mask = labels == cluster_id
            texts = df.loc[mask, '_document_text'].tolist()

            labeler = ClusterLabeler(llm_provider=st.session_state.llm_provider)
            previous_label = st.session_state.cluster_info.get(cluster_id, {}).get('label', '')

            new_info = labeler.regenerate_label(cluster_id, texts, previous_label)
            st.session_state.cluster_info[cluster_id] = new_info

            st.success(f"新しいラベル: {new_info['label']}")
            st.rerun()

        except Exception as e:
            st.error(f"再生成エラー: {e}")


def main():
    """メイン関数"""
    init_session_state()

    st.title("テキストクラスタリング")
    st.caption("Embedding → UMAP → HDBSCAN → ラベリング → 可視化")

    # サイドバー
    settings = render_sidebar()

    # タブ
    tab1, tab2, tab3 = st.tabs([
        "データ取込/成形",
        "可視化",
        "クラスタ一覧レポート"
    ])

    with tab1:
        render_tab_import()

    with tab2:
        render_tab_visualization(settings)

    with tab3:
        render_tab_report()


if __name__ == "__main__":
    main()
