"""
可視化モジュール
Plotlyを使用した散布図・チャート生成
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def create_scatter_plot(
    df: pd.DataFrame,
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    cluster_info: Optional[Dict[int, Dict]] = None,
    highlight_cluster: Optional[int] = None,
    show_labels: bool = True,
    title: str = "クラスタリング結果（UMAP 2D）",
    max_text_length: int = 100
) -> go.Figure:
    """
    UMAP 2D散布図を作成

    Args:
        df: 元のDataFrame
        embeddings_2d: 2D座標 (N, 2)
        labels: クラスタラベル配列
        cluster_info: クラスタ情報辞書（ラベル名など）
        highlight_cluster: ハイライトするクラスタID（Noneで全表示）
        show_labels: グラフ内にクラスタ名ラベルを表示するか
        title: グラフタイトル
        max_text_length: ホバーテキストの最大長

    Returns:
        go.Figure: Plotlyフィギュア
    """
    # DataFrameにUMAP座標とクラスタを追加
    plot_df = df.copy()
    plot_df['x'] = embeddings_2d[:, 0]
    plot_df['y'] = embeddings_2d[:, 1]
    plot_df['cluster'] = labels

    # クラスタラベル名を追加（切り詰めなし）
    if cluster_info:
        plot_df['cluster_label'] = plot_df['cluster'].apply(
            lambda c: cluster_info.get(c, {}).get('label', f'クラスタ {c}')
        )
    else:
        plot_df['cluster_label'] = plot_df['cluster'].apply(
            lambda c: 'ノイズ' if c == -1 else f'クラスタ {c}'
        )

    # ホバーテキスト用に短縮（ホバーのみ）
    def truncate(text, max_len):
        if pd.isna(text) or not text:
            return ""
        text = str(text)
        return text[:max_len] + "..." if len(text) > max_len else text

    plot_df['概要_short'] = plot_df.get('取り組み宣言（概要）', pd.Series([''] * len(plot_df))).apply(
        lambda x: truncate(x, max_text_length)
    )
    plot_df['詳細_short'] = plot_df.get('取り組み宣言（詳細）', pd.Series([''] * len(plot_df))).apply(
        lambda x: truncate(x, max_text_length)
    )

    # 透明度設定（ハイライト時）
    if highlight_cluster is not None:
        plot_df['opacity'] = plot_df['cluster'].apply(
            lambda c: 1.0 if c == highlight_cluster else 0.2
        )
    else:
        plot_df['opacity'] = 1.0

    # カラーパレット（ノイズは灰色）
    unique_clusters = sorted(plot_df['cluster'].unique())
    n_clusters = len([c for c in unique_clusters if c != -1])

    # カスタムカラースケール
    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    cluster_colors = {}
    color_idx = 0
    for c in unique_clusters:
        if c == -1:
            cluster_colors[c] = 'rgba(128, 128, 128, 0.5)'
        else:
            cluster_colors[c] = colors[color_idx % len(colors)]
            color_idx += 1

    plot_df['color'] = plot_df['cluster'].map(cluster_colors)

    # Plotlyでプロット
    fig = go.Figure()

    # 各クラスタの中心座標を計算
    cluster_centers = {}
    for cluster_id in unique_clusters:
        cluster_data = plot_df[plot_df['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            center_x = cluster_data['x'].mean()
            center_y = cluster_data['y'].mean()
            cluster_centers[cluster_id] = (center_x, center_y)

    for cluster_id in unique_clusters:
        cluster_data = plot_df[plot_df['cluster'] == cluster_id]

        hover_template = (
            "<b>%{customdata[0]}</b><br>"
            "ブロック: %{customdata[1]}<br>"
            "カテゴリ: %{customdata[2]}<br>"
            "概要: %{customdata[3]}<br>"
            "詳細: %{customdata[4]}<br>"
            "<extra></extra>"
        )

        customdata = np.column_stack([
            cluster_data.get('店舗名', pd.Series([''] * len(cluster_data))),
            cluster_data.get('ブロック番号', pd.Series([''] * len(cluster_data))),
            cluster_data.get('カテゴリ名', pd.Series([''] * len(cluster_data))),
            cluster_data['概要_short'],
            cluster_data['詳細_short']
        ])

        # フルラベル名（切り詰めなし）
        label_name = cluster_data['cluster_label'].iloc[0] if len(cluster_data) > 0 else f'クラスタ {cluster_id}'

        opacity = 1.0 if highlight_cluster is None else (1.0 if cluster_id == highlight_cluster else 0.2)

        fig.add_trace(go.Scatter(
            x=cluster_data['x'],
            y=cluster_data['y'],
            mode='markers',
            name=label_name,
            marker=dict(
                size=8,
                color=cluster_colors[cluster_id],
                opacity=opacity,
                line=dict(width=0.5, color='white')
            ),
            customdata=customdata,
            hovertemplate=hover_template
        ))

    # クラスタ名をグラフエリア内に表示（各クラスタの中心に□で囲んだラベル）
    annotations = []
    if show_labels:
        for cluster_id in unique_clusters:
            if cluster_id in cluster_centers:
                center_x, center_y = cluster_centers[cluster_id]

                if cluster_info and cluster_id in cluster_info:
                    label_text = cluster_info[cluster_id].get('label', f'クラスタ {cluster_id}')
                else:
                    label_text = 'ノイズ' if cluster_id == -1 else f'クラスタ {cluster_id}'

                # ハイライト時は対象クラスタのみ表示
                if highlight_cluster is not None and cluster_id != highlight_cluster:
                    continue

                annotations.append(dict(
                    x=center_x,
                    y=center_y,
                    text=label_text,
                    showarrow=False,
                    font=dict(size=11, color='black', family='sans-serif'),
                    bgcolor='rgba(255, 255, 255, 0.85)',
                    bordercolor=cluster_colors.get(cluster_id, 'gray'),
                    borderwidth=2,
                    borderpad=4,
                ))

    fig.update_layout(
        title=title,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="クラスタ",
        hovermode='closest',
        height=600,
        template='plotly_white',
        annotations=annotations
    )

    return fig


def create_category_distribution_chart(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_info: Optional[Dict[int, Dict]] = None
) -> go.Figure:
    """
    カテゴリ分布チャートを作成

    Args:
        df: DataFrame
        labels: クラスタラベル
        cluster_info: クラスタ情報

    Returns:
        go.Figure: Plotlyフィギュア
    """
    plot_df = df.copy()
    plot_df['cluster'] = labels

    if cluster_info:
        plot_df['cluster_label'] = plot_df['cluster'].apply(
            lambda c: cluster_info.get(c, {}).get('label', f'クラスタ {c}')
        )
    else:
        plot_df['cluster_label'] = plot_df['cluster'].apply(
            lambda c: 'ノイズ' if c == -1 else f'クラスタ {c}'
        )

    # カテゴリ × クラスタのクロス集計
    if 'カテゴリ名' in plot_df.columns:
        cross_tab = pd.crosstab(plot_df['cluster_label'], plot_df['カテゴリ名'])

        fig = px.bar(
            cross_tab,
            barmode='stack',
            title="クラスタ別カテゴリ分布"
        )

        fig.update_layout(
            xaxis_title="クラスタ",
            yaxis_title="件数",
            legend_title="カテゴリ",
            height=400
        )
    else:
        # カテゴリがない場合はクラスタ件数のみ
        cluster_counts = plot_df['cluster_label'].value_counts()

        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="クラスタ別件数"
        )

        fig.update_layout(
            xaxis_title="クラスタ",
            yaxis_title="件数",
            height=400
        )

    return fig


def create_cluster_size_pie(
    labels: np.ndarray,
    cluster_info: Optional[Dict[int, Dict]] = None
) -> go.Figure:
    """
    クラスタサイズの円グラフを作成

    Args:
        labels: クラスタラベル
        cluster_info: クラスタ情報

    Returns:
        go.Figure: Plotlyフィギュア
    """
    from collections import Counter

    counter = Counter(labels)

    cluster_ids = []
    cluster_names = []
    counts = []

    for cluster_id, count in sorted(counter.items()):
        cluster_ids.append(cluster_id)
        if cluster_info and cluster_id in cluster_info:
            name = cluster_info[cluster_id].get('label', f'クラスタ {cluster_id}')
        else:
            name = 'ノイズ' if cluster_id == -1 else f'クラスタ {cluster_id}'
        cluster_names.append(name)
        counts.append(count)

    fig = px.pie(
        values=counts,
        names=cluster_names,
        title="クラスタ構成比"
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)

    return fig


def create_store_cluster_heatmap(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_info: Optional[Dict[int, Dict]] = None
) -> go.Figure:
    """
    店舗×クラスタのヒートマップを作成

    Args:
        df: DataFrame
        labels: クラスタラベル
        cluster_info: クラスタ情報

    Returns:
        go.Figure: Plotlyフィギュア
    """
    plot_df = df.copy()
    plot_df['cluster'] = labels

    if cluster_info:
        plot_df['cluster_label'] = plot_df['cluster'].apply(
            lambda c: cluster_info.get(c, {}).get('label', f'クラスタ {c}')
        )
    else:
        plot_df['cluster_label'] = plot_df['cluster'].apply(
            lambda c: 'ノイズ' if c == -1 else f'クラスタ {c}'
        )

    if '店舗名' not in plot_df.columns:
        return go.Figure()

    # クロス集計
    cross_tab = pd.crosstab(plot_df['店舗名'], plot_df['cluster_label'])

    fig = px.imshow(
        cross_tab,
        title="店舗×クラスタ ヒートマップ",
        labels=dict(x="クラスタ", y="店舗名", color="件数"),
        color_continuous_scale="Blues"
    )

    fig.update_layout(height=max(400, len(cross_tab) * 25))

    return fig


def export_to_csv(
    df: pd.DataFrame,
    labels: Optional[np.ndarray] = None,
    cluster_info: Optional[Dict[int, Dict]] = None,
    include_cluster: bool = True
) -> str:
    """
    DataFrameをCSV文字列にエクスポート

    Args:
        df: DataFrame
        labels: クラスタラベル（Noneの場合はクラスタ列なし）
        cluster_info: クラスタ情報（ラベル名用）
        include_cluster: クラスタ列を含めるか

    Returns:
        str: CSV文字列
    """
    export_df = df.copy()

    # 内部列を除外
    internal_cols = [c for c in export_df.columns if c.startswith('_')]
    export_df = export_df.drop(columns=internal_cols, errors='ignore')

    if include_cluster and labels is not None:
        export_df['クラスタ番号'] = labels

        if cluster_info:
            export_df['クラスタ名'] = pd.Series(labels).apply(
                lambda c: cluster_info.get(c, {}).get('label', f'クラスタ {c}')
            ).values

    return export_df.to_csv(index=False, encoding='utf-8-sig')
