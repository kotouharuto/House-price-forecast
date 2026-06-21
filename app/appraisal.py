"""物件査定ページのドメインロジック（Streamlit 非依存・テスト可能）.

サイドバーのウィジェット呼び出しから分離した純粋関数群。候補物件の絞り込みと
表示ラベル生成を提供し、UI（``app/pages/4_物件査定.py``）はこれらを呼び出すだけにする。
スライダー操作＝範囲条件の適用なので、``apply_range`` / ``apply_filters`` を
テストすればフィルタ挙動を検証できる。
"""

import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.visualization.aggregate import (  # noqa: E402
    AGE_COL,
    AREA_COL,
    STATION_DISTANCE_COL,
    TYPE_COL,
    WARD_CODE_COL,
)
from src.visualization.format import format_yen_jp  # noqa: E402

# features.csv 上の取引価格列（学習用特徴量とは別に表示・絞り込みで使う）
PRICE_COL = "取引価格（総額）"


def build_candidate_pool(
    df: pd.DataFrame,
    ward_code: int,
    property_type: str,
) -> pd.DataFrame:
    """行政区コードと種類で査定候補の母集団を抽出する.

    Args:
        df: 全取引データ。
        ward_code: 市区町村コード。
        property_type: 物件種類（``種類`` 列の値）。

    Returns:
        条件に一致する候補 DataFrame。
    """
    return df[(df[WARD_CODE_COL] == ward_code) & (df[TYPE_COL] == property_type)]


def apply_range(candidates: pd.DataFrame, column: str, low: float, high: float) -> pd.DataFrame:
    """指定列の値が ``[low, high]``（両端含む）に入る行のみを返す.

    スライダーを ``(low, high)`` に動かした状態に相当する。欠損値を持つ行は
    範囲条件を満たさないため除外される。

    Args:
        candidates: 絞り込み対象。
        column: 範囲条件を適用する数値列。
        low: 範囲の下限（含む）。
        high: 範囲の上限（含む）。

    Returns:
        範囲条件を満たす行のみの DataFrame。
    """
    return candidates[candidates[column].between(low, high)]


def apply_filters(
    candidates: pd.DataFrame,
    ranges: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    """複数列の範囲条件をまとめて適用する.

    Args:
        candidates: 絞り込み対象。
        ranges: ``{列名: (下限, 上限)}`` の対応。

    Returns:
        すべての範囲条件を満たす行の DataFrame。
    """
    result = candidates
    for column, (low, high) in ranges.items():
        result = apply_range(result, column, low, high)
    return result


def property_label(row: pd.Series) -> str:
    """セレクトボックス用に物件1件を1行の文字列で表す."""
    return (
        f"{row[AREA_COL]:.0f}㎡ / 築{row[AGE_COL]:.0f}年 / "
        f"駅{row[STATION_DISTANCE_COL]:.0f}分 / {format_yen_jp(row[PRICE_COL])}"
    )
