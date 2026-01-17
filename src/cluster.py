"""
クラスタリングモジュール
Embedding → UMAP → HDBSCAN
ダミーモード: TF-IDF + KMeans

Note: sklearn.cluster.HDBSCAN を使用（Python 3.14対応）
      外部hdbscanパッケージは不要
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class ClusteringResult:
    """クラスタリング結果を保持するクラス"""

    def __init__(
        self,
        labels: np.ndarray,
        embeddings_2d: np.ndarray,
        embeddings_original: Optional[np.ndarray] = None,
        method: str = "hdbscan",
        n_clusters: int = 0,
        n_noise: int = 0,
        silhouette: Optional[float] = None
    ):
        self.labels = labels
        self.embeddings_2d = embeddings_2d
        self.embeddings_original = embeddings_original
        self.method = method
        self.n_clusters = n_clusters
        self.n_noise = n_noise
        self.silhouette = silhouette

    def to_dict(self) -> Dict:
        return {
            'method': self.method,
            'n_clusters': self.n_clusters,
            'n_noise': self.n_noise,
            'silhouette': self.silhouette,
            'labels': self.labels.tolist(),
        }


class TextClusterer:
    """テキストクラスタリングクラス"""

    def __init__(
        self,
        use_api: bool = True,
        llm_provider=None,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: int = 3,
        kmeans_n_clusters: int = 8,
        random_state: int = 42
    ):
        self.use_api = use_api
        self.llm_provider = llm_provider
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.kmeans_n_clusters = kmeans_n_clusters
        self.random_state = random_state

        self._embeddings = None
        self._embeddings_2d = None
        self._labels = None

    def _get_embeddings_api(self, texts: List[str]) -> np.ndarray:
        """LLM APIを使用して埋め込みを取得"""
        if not self.llm_provider:
            raise ValueError("LLMプロバイダが設定されていません")

        if not self.llm_provider.supports_embedding:
            raise ValueError(
                f"{self.llm_provider.name} は埋め込みAPIをサポートしていません。"
                "OpenAI または Gemini のAPIキーを使用してください。"
            )

        embeddings = self.llm_provider.embed(texts)
        return np.array(embeddings)

    def _get_embeddings_tfidf(self, texts: List[str]) -> np.ndarray:
        """TF-IDFを使用して埋め込みを取得（ダミーモード）"""
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        # 空のテキストを処理
        processed_texts = [t if t.strip() else " " for t in texts]

        try:
            embeddings = vectorizer.fit_transform(processed_texts)
            return embeddings.toarray()
        except ValueError as e:
            # min_dfなどで全て除外された場合
            logger.warning(f"TF-IDF failed with error: {e}, using simpler settings")
            vectorizer = TfidfVectorizer(max_features=500)
            embeddings = vectorizer.fit_transform(processed_texts)
            return embeddings.toarray()

    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """UMAPで2次元に次元削減"""
        try:
            import umap

            n_samples = embeddings.shape[0]
            n_neighbors = min(self.umap_n_neighbors, n_samples - 1)

            if n_neighbors < 2:
                logger.warning("サンプル数が少なすぎるため、UMAPの代わりにPCAを使用")
                return self._reduce_dimensions_pca(embeddings)

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=self.umap_min_dist,
                metric='cosine',
                random_state=self.random_state
            )

            embeddings_2d = reducer.fit_transform(embeddings)
            return embeddings_2d

        except Exception as e:
            logger.warning(f"UMAP failed: {e}, falling back to PCA")
            return self._reduce_dimensions_pca(embeddings)

    def _reduce_dimensions_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """PCAで2次元に次元削減（フォールバック）"""
        from sklearn.decomposition import PCA

        n_components = min(2, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n_components, random_state=self.random_state)
        embeddings_2d = pca.fit_transform(embeddings)

        if embeddings_2d.shape[1] == 1:
            embeddings_2d = np.column_stack([embeddings_2d, np.zeros(len(embeddings_2d))])

        return embeddings_2d

    def _cluster_hdbscan(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """HDBSCANでクラスタリング（sklearn.cluster.HDBSCAN使用）"""
        try:
            n_samples = embeddings.shape[0]
            min_cluster_size = min(self.hdbscan_min_cluster_size, max(2, n_samples // 5))
            min_samples = min(self.hdbscan_min_samples, min_cluster_size)

            # sklearn.cluster.HDBSCAN を使用（Python 3.14対応）
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_epsilon=0.0,
                cluster_selection_method='eom'
            )

            labels = clusterer.fit_predict(embeddings)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

            # ノイズが多すぎる場合（50%以上）はKMeansにフォールバック
            if n_noise > len(labels) * 0.5:
                logger.warning(f"HDBSCAN produced too much noise ({n_noise}/{len(labels)}), falling back to KMeans")
                return self._cluster_kmeans(embeddings)

            return labels, n_clusters, n_noise

        except Exception as e:
            logger.warning(f"HDBSCAN failed: {e}, falling back to KMeans")
            return self._cluster_kmeans(embeddings)

    def _cluster_kmeans(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """KMeansでクラスタリング（指定されたクラスタ数を使用）"""
        n_samples = embeddings.shape[0]
        # 指定されたクラスタ数を使用（サンプル数を超えない範囲で）
        n_clusters = min(self.kmeans_n_clusters, max(2, n_samples - 1))

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(embeddings)

        return labels, n_clusters, 0

    def fit(self, texts: List[str], use_hdbscan: bool = True) -> ClusteringResult:
        """
        テキストをクラスタリング

        Args:
            texts: クラスタリング対象のテキストリスト
            use_hdbscan: Trueの場合HDBSCAN、Falseの場合KMeansを使用

        Returns:
            ClusteringResult: クラスタリング結果
        """
        if not texts:
            raise ValueError("テキストが空です")

        # 1. 埋め込み取得
        logger.info(f"Getting embeddings for {len(texts)} texts...")
        if self.use_api and self.llm_provider:
            self._embeddings = self._get_embeddings_api(texts)
        else:
            self._embeddings = self._get_embeddings_tfidf(texts)

        # 2. 次元削減
        logger.info("Reducing dimensions with UMAP...")
        self._embeddings_2d = self._reduce_dimensions(self._embeddings)

        # 3. クラスタリング
        logger.info(f"Clustering with {'HDBSCAN' if use_hdbscan else 'KMeans'}...")
        if use_hdbscan:
            labels, n_clusters, n_noise = self._cluster_hdbscan(self._embeddings_2d)
            method = "hdbscan"
        else:
            labels, n_clusters, n_noise = self._cluster_kmeans(self._embeddings_2d)
            method = "kmeans"

        self._labels = labels

        # 4. シルエットスコア計算
        silhouette = None
        if n_clusters > 1 and len(set(labels)) > 1:
            try:
                # ノイズラベル(-1)を除外してスコア計算
                valid_mask = labels >= 0
                if valid_mask.sum() > n_clusters:
                    silhouette = silhouette_score(
                        self._embeddings_2d[valid_mask],
                        labels[valid_mask]
                    )
            except Exception as e:
                logger.warning(f"Silhouette score calculation failed: {e}")

        result = ClusteringResult(
            labels=labels,
            embeddings_2d=self._embeddings_2d,
            embeddings_original=self._embeddings,
            method=method,
            n_clusters=n_clusters,
            n_noise=n_noise,
            silhouette=silhouette
        )

        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")
        return result

    def get_cluster_samples(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        n_samples: int = 3
    ) -> Dict[int, pd.DataFrame]:
        """
        各クラスタから代表サンプルを取得

        Args:
            df: 元のDataFrame
            labels: クラスタラベル
            n_samples: 各クラスタから取得するサンプル数

        Returns:
            Dict[int, pd.DataFrame]: クラスタID -> サンプルDataFrame
        """
        cluster_samples = {}

        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            cluster_df = df[mask]

            # クラスタ内からランダムにサンプリング
            n = min(n_samples, len(cluster_df))
            samples = cluster_df.sample(n=n, random_state=self.random_state)
            cluster_samples[cluster_id] = samples

        return cluster_samples

    def get_cluster_stats(
        self,
        df: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[int, Dict]:
        """
        各クラスタの統計情報を取得

        Args:
            df: 元のDataFrame
            labels: クラスタラベル

        Returns:
            Dict[int, Dict]: クラスタID -> 統計情報
        """
        stats = {}

        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            cluster_df = df[mask]

            cluster_stats = {
                'count': len(cluster_df),
                'store_count': cluster_df['店舗名'].nunique() if '店舗名' in cluster_df else 0,
                'category_dist': {},
            }

            # カテゴリ分布
            if 'カテゴリ名' in cluster_df:
                cluster_stats['category_dist'] = cluster_df['カテゴリ名'].value_counts().to_dict()

            stats[cluster_id] = cluster_stats

        return stats
