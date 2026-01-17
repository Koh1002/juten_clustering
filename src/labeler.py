"""
ラベル生成モジュール
- LLMによるクラスタラベル生成
- 頻出語抽出
- 特徴説明生成
"""

import re
from collections import Counter
from typing import List, Dict, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# 日本語ストップワード（簡易版）
JAPANESE_STOPWORDS = {
    'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ',
    'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や',
    'など', 'なっ', 'ない', 'この', 'ため', 'その', 'あっ', 'よう', 'また',
    'もの', 'という', 'あり', 'まで', 'られ', 'なる', 'へ', 'か', 'だ', 'これ',
    'によって', 'により', 'おり', 'より', 'による', 'ず', 'なり', 'られる',
    'において', 'ば', 'なかっ', 'なく', 'しかし', 'について', 'せ', 'だっ',
    'その他', '及び', 'できる', 'できない', '等', 'ほか', '以上', '以下',
    '場合', '必要', '対応', '実施', '実現', '推進', '取り組み', '取組',
    '目標', '達成', '向上', '強化', '徹底', '継続', '改善', '活動',
    'ます', 'です', 'ました', 'でした', 'ください', 'します', 'しました',
}


class ClusterLabeler:
    """クラスタラベル生成クラス"""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.cluster_info: Dict[int, Dict] = {}

    def extract_frequent_words(
        self,
        texts: List[str],
        top_n: int = 10,
        min_word_length: int = 2
    ) -> List[tuple]:
        """
        テキストから頻出語を抽出

        Args:
            texts: テキストリスト
            top_n: 上位何件を返すか
            min_word_length: 最小単語長

        Returns:
            List[tuple]: (単語, 出現回数) のリスト
        """
        try:
            from janome.tokenizer import Tokenizer
            tokenizer = Tokenizer()
            use_janome = True
        except ImportError:
            logger.warning("Janome not available, using simple tokenization")
            use_janome = False

        word_counter = Counter()

        for text in texts:
            if not text or not text.strip():
                continue

            if use_janome:
                # Janomeで形態素解析
                tokens = tokenizer.tokenize(text)
                for token in tokens:
                    # 名詞、動詞、形容詞のみ抽出
                    pos = token.part_of_speech.split(',')[0]
                    if pos in ['名詞', '動詞', '形容詞']:
                        word = token.base_form
                        if (
                            len(word) >= min_word_length and
                            word not in JAPANESE_STOPWORDS and
                            not word.isdigit() and
                            not re.match(r'^[a-zA-Z]+$', word)
                        ):
                            word_counter[word] += 1
            else:
                # 簡易トークン化（空白とカタカナ・漢字の境界で分割）
                words = re.findall(r'[一-龥ぁ-んァ-ン]+', text)
                for word in words:
                    if (
                        len(word) >= min_word_length and
                        word not in JAPANESE_STOPWORDS
                    ):
                        word_counter[word] += 1

        return word_counter.most_common(top_n)

    def generate_cluster_label(
        self,
        cluster_id: int,
        texts: List[str],
        n_sample: int = 5
    ) -> Dict:
        """
        クラスタのラベルと説明を生成

        Args:
            cluster_id: クラスタID
            texts: クラスタ内のテキストリスト
            n_sample: プロンプトに含めるサンプル数

        Returns:
            Dict: ラベル情報
        """
        # 頻出語抽出
        frequent_words = self.extract_frequent_words(texts)
        word_list = [w[0] for w in frequent_words[:10]]

        # サンプルテキスト
        sample_texts = texts[:n_sample] if len(texts) > n_sample else texts
        sample_texts = [t[:200] for t in sample_texts]  # 長さ制限

        # ラベルがノイズ（-1）の場合
        if cluster_id == -1:
            return {
                'cluster_id': cluster_id,
                'label': '未分類（ノイズ）',
                'description': 'HDBSCANによってクラスタに属さないと判定されたデータポイントです。',
                'representative_texts': sample_texts[:3],
                'frequent_words': word_list,
                'count': len(texts)
            }

        # LLMが利用可能な場合
        if self.llm_provider:
            try:
                label, description = self._generate_with_llm(
                    cluster_id, sample_texts, word_list
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                label, description = self._generate_fallback(
                    cluster_id, sample_texts, word_list
                )
        else:
            label, description = self._generate_fallback(
                cluster_id, sample_texts, word_list
            )

        result = {
            'cluster_id': cluster_id,
            'label': label,
            'description': description,
            'representative_texts': sample_texts[:3],
            'frequent_words': word_list,
            'count': len(texts)
        }

        self.cluster_info[cluster_id] = result
        return result

    def _generate_with_llm(
        self,
        cluster_id: int,
        sample_texts: List[str],
        frequent_words: List[str]
    ) -> tuple:
        """LLMでラベルと説明を生成"""
        samples_str = "\n".join([f"- {t}" for t in sample_texts])
        words_str = ", ".join(frequent_words)

        prompt = f"""以下は、あるテキストクラスタに含まれるサンプルです。

【サンプルテキスト】
{samples_str}

【頻出語】
{words_str}

このクラスタの内容を分析し、以下の形式で回答してください：

【ラベル】
（このクラスタを表す日本語ラベル、20〜30文字程度で具体的に）

【特徴説明】
（このクラスタの特徴を3〜5行で説明）

回答："""

        response = self.llm_provider.generate(prompt, max_tokens=500)

        # レスポンスをパース
        label = f"クラスタ {cluster_id}"
        description = ""

        label_match = re.search(r'【ラベル】\s*(.+?)(?=【|$)', response, re.DOTALL)
        if label_match:
            # ラベルは切り詰めずにそのまま使用
            label = label_match.group(1).strip()

        desc_match = re.search(r'【特徴説明】\s*(.+?)(?=【|$)', response, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()

        if not description:
            description = response.strip()[:500]

        return label, description

    def _generate_fallback(
        self,
        cluster_id: int,
        sample_texts: List[str],
        frequent_words: List[str]
    ) -> tuple:
        """フォールバック：頻出語からラベルを生成"""
        if frequent_words:
            # 頻出語を組み合わせてラベルを作成（切り詰めなし）
            if len(frequent_words) >= 2:
                label = f"{frequent_words[0]}・{frequent_words[1]}関連"
            else:
                label = f"{frequent_words[0]}関連"
            words_desc = "、".join(frequent_words[:5])
            description = f"このクラスタは「{words_desc}」などのキーワードを含むテキストで構成されています。"
        else:
            label = f"クラスタ {cluster_id}"
            description = "このクラスタの特徴を分析中です。"

        return label, description

    def regenerate_label(
        self,
        cluster_id: int,
        texts: List[str],
        previous_label: str = ""
    ) -> Dict:
        """
        「別アイデア！」ボタン用：ラベルのみ再生成

        Args:
            cluster_id: クラスタID
            texts: クラスタ内のテキスト
            previous_label: 前回のラベル

        Returns:
            Dict: 新しいラベル情報
        """
        if not self.llm_provider:
            return self.cluster_info.get(cluster_id, {})

        # 頻出語抽出
        frequent_words = self.extract_frequent_words(texts)
        word_list = [w[0] for w in frequent_words[:10]]

        sample_texts = texts[:5] if len(texts) > 5 else texts
        sample_texts = [t[:200] for t in sample_texts]
        samples_str = "\n".join([f"- {t}" for t in sample_texts])
        words_str = ", ".join(word_list)

        prompt = f"""以下のテキストクラスタに、新しいラベルを付けてください。
前回のラベル「{previous_label}」とは異なる視点でお願いします。

【サンプルテキスト】
{samples_str}

【頻出語】
{words_str}

新しいラベル（20〜30文字程度の日本語で具体的に）："""

        try:
            response = self.llm_provider.generate(prompt, max_tokens=100)
            # ラベルは切り詰めずにそのまま使用
            new_label = response.strip()
        except Exception as e:
            logger.error(f"Label regeneration failed: {e}")
            new_label = f"クラスタ {cluster_id} (v2)"

        # 既存情報を更新
        if cluster_id in self.cluster_info:
            self.cluster_info[cluster_id]['label'] = new_label
            return self.cluster_info[cluster_id]

        return {
            'cluster_id': cluster_id,
            'label': new_label,
            'description': '',
            'representative_texts': sample_texts[:3],
            'frequent_words': word_list,
            'count': len(texts)
        }

    def generate_all_labels(
        self,
        df: pd.DataFrame,
        labels: 'np.ndarray',
        text_column: str = '_document_text'
    ) -> Dict[int, Dict]:
        """
        全クラスタのラベルを生成

        Args:
            df: DataFrame
            labels: クラスタラベル配列
            text_column: テキスト列名

        Returns:
            Dict[int, Dict]: クラスタID -> ラベル情報
        """
        import numpy as np

        self.cluster_info = {}

        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            cluster_texts = df.loc[mask, text_column].tolist()

            logger.info(f"Generating label for cluster {cluster_id} ({len(cluster_texts)} texts)")

            label_info = self.generate_cluster_label(cluster_id, cluster_texts)
            self.cluster_info[cluster_id] = label_info

        return self.cluster_info

    def get_summary_stats(
        self,
        df: pd.DataFrame,
        labels: 'np.ndarray'
    ) -> Dict:
        """
        全体サマリ統計を取得

        Args:
            df: DataFrame
            labels: クラスタラベル配列

        Returns:
            Dict: サマリ統計
        """
        import numpy as np

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = (labels == -1).sum()

        # カテゴリ別件数
        category_counts = {}
        if 'カテゴリ名' in df.columns:
            category_counts = df['カテゴリ名'].value_counts().to_dict()

        # 店舗別件数
        store_counts = {}
        if '店舗名' in df.columns:
            store_counts = df['店舗名'].value_counts().to_dict()

        return {
            'total_records': len(df),
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(df) if len(df) > 0 else 0,
            'category_counts': category_counts,
            'store_counts': store_counts,
            'unique_stores': len(store_counts),
        }
