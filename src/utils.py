"""
共通ユーティリティ関数
"""

import re
import logging
from typing import Optional

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_text(text: Optional[str]) -> str:
    """
    テキストを正規化する
    - 前後空白削除
    - 連続空白を1つに圧縮
    - 改行を半角スペースに変換
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 改行を空白に変換
    text = text.replace('\n', ' ').replace('\r', ' ')
    # 連続空白を1つに圧縮
    text = re.sub(r'\s+', ' ', text)
    # 前後空白削除
    text = text.strip()

    return text


def truncate_text(text: str, max_length: int = 100) -> str:
    """テキストを指定長で切り詰め"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def normalize_number(text: str) -> str:
    """全角数字を半角に変換"""
    trans_table = str.maketrans('０１２３４５６７８９', '0123456789')
    return text.translate(trans_table)


def is_summary_row(value: Optional[str]) -> bool:
    """
    概要行かどうか判定
    E列が正規表現 `^\s*[0-9０-９]+[\.．]\s*$` に一致
    """
    if value is None:
        return False
    if not isinstance(value, str):
        value = str(value)

    pattern = r'^\s*[0-9０-９]+[\.．]\s*$'
    return bool(re.match(pattern, value))


def is_detail_row(value: Optional[str]) -> bool:
    """
    詳細行かどうか判定
    F列が正規表現 `^\s*\([0-9０-９]+\)\s*$` に一致
    """
    if value is None:
        return False
    if not isinstance(value, str):
        value = str(value)

    # 全角括弧も対応
    pattern = r'^\s*[\(（][0-9０-９]+[\)）]\s*$'
    return bool(re.match(pattern, value))


def extract_category(value: Optional[str]) -> Optional[str]:
    """
    カテゴリ値を抽出
    A〜E または 【A】〜【E】 形式に対応
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    # 【A】形式
    match = re.match(r'【([A-Ea-e])】', value)
    if match:
        return match.group(1).upper()

    # 単独A〜E
    if re.match(r'^[A-Ea-e]$', value):
        return value.upper()

    return None


def get_category_name(category: Optional[str]) -> str:
    """カテゴリ記号からカテゴリ名を取得"""
    mapping = {
        'A': '使命',
        'B': '利益',
        'C': '能力開発',
        'D': '実現力',
        'E': 'コストコントロール'
    }
    if category is None:
        return '未分類'
    return mapping.get(category.upper(), '未分類')


def join_non_empty_cells(cells: list) -> str:
    """
    空でないセルをスペース結合
    """
    result = []
    for cell in cells:
        if cell is not None and str(cell).strip():
            result.append(normalize_text(str(cell)))
    return ' '.join(result)


def calculate_category_column_score(values: list) -> dict:
    """
    カテゴリ列候補のスコアを計算
    - マッチ率（A〜Eにマッチする比率）
    - 非空率
    """
    total = len(values)
    if total == 0:
        return {'match_rate': 0, 'non_empty_rate': 0, 'score': 0, 'examples': []}

    non_empty = 0
    matched = 0
    examples = []

    for v in values:
        if v is not None and str(v).strip():
            non_empty += 1
            cat = extract_category(str(v))
            if cat:
                matched += 1
                if len(examples) < 5:
                    examples.append(str(v).strip())

    match_rate = matched / total if total > 0 else 0
    non_empty_rate = non_empty / total if total > 0 else 0

    # スコア: マッチ率を重視しつつ、非空率も考慮
    score = match_rate * 0.8 + non_empty_rate * 0.2

    return {
        'match_rate': match_rate,
        'non_empty_rate': non_empty_rate,
        'score': score,
        'examples': examples,
        'matched_count': matched,
        'non_empty_count': non_empty,
        'total': total
    }


def create_document_text(category_name: str, summary: str, detail: str) -> str:
    """
    クラスタリング用の文書テキストを生成
    """
    parts = [category_name]
    if summary:
        parts.append(summary)
    if detail:
        parts.append(detail)
    return ' / '.join(parts)
