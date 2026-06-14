"""類似物件比較（取引事例比較）モジュール.

対象物件に「似た過去取引」を抽出して返す純粋関数を提供する。
階層フィルタ（種類・市区町村コード）で母集団を絞った後、面積・築年数・
最寄駅距離の正規化ユークリッド距離で kNN を行う。

不動産鑑定における「取引事例比較法」のアナロジーで、AVM 予測値の
妥当性を**事実データ（過去の実取引）**で裏付けるための関数群。
モデルの予測ではなく、対象に近い実取引を示すことで、因果との誤読を避けつつ
顧客に説得力のある説明を行うことを目的とする。
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 既定の階層フィルタ列（同じ種類・同じ市区町村に限定する）
_DEFAULT_FILTER_COLUMNS: tuple[str, ...] = ("種類", "市区町村コード")

# 既定の距離計算用連続特徴量（正規化ユークリッド距離の対象）
_DEFAULT_DISTANCE_FEATURES: tuple[str, ...] = (
    "面積（㎡）",
    "築年数",
    "最寄駅：距離（分）",
)

# 取引時期の並び替えに使う列（降順 = 直近優先）
_RECENCY_COLUMNS: tuple[str, ...] = ("取引年", "取引四半期")


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str], purpose: str) -> None:
    """必須列の存在を検証し、欠落していれば KeyError を送出する."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"類似物件検索に必要な列が不足しています（{purpose}）: {missing}")


def _build_pool(
    df: pd.DataFrame,
    target_row: pd.Series,
    filter_columns: Sequence[str],
) -> pd.DataFrame:
    """対象物件と同じカテゴリ属性を持つ母集団を抽出する.

    Args:
        df: 全取引データ。
        target_row: 対象物件の1行。
        filter_columns: 完全一致フィルタに使う列。

    Returns:
        フィルタ後の DataFrame。
    """
    mask = pd.Series(True, index=df.index)
    for col in filter_columns:
        mask &= df[col] == target_row[col]
    return df.loc[mask]


def _compute_distance(
    pool: pd.DataFrame,
    target_row: pd.Series,
    distance_features: Sequence[str],
) -> pd.Series:
    """対象物件と母集団との正規化ユークリッド距離を計算する.

    各特徴量を母集団の標準偏差で割って正規化することで、スケールの異なる
    特徴量（面積と駅距離など）を同等に扱う。

    Args:
        pool: フィルタ後の母集団。
        target_row: 対象物件の1行。
        distance_features: 距離計算に使う連続特徴量。

    Returns:
        ``pool`` の index に対応する距離の Series。距離計算列に欠損が
        ある行は NaN を返す（呼び出し元で除外）。
    """
    feature_matrix = pool[list(distance_features)].astype(float)
    target_vec = target_row[list(distance_features)].astype(float).to_numpy()

    # 母集団の標準偏差で正規化（z-score 的）。0 や NaN の列は重みを 0 にする。
    std = feature_matrix.std(ddof=0).to_numpy()
    safe_std = np.where((std == 0) | np.isnan(std), 1.0, std)

    diff = (feature_matrix.to_numpy() - target_vec) / safe_std
    # 各行のいずれかが NaN なら距離は NaN になる
    squared = np.square(diff).sum(axis=1)
    return pd.Series(np.sqrt(squared), index=pool.index)


def find_similar_properties(
    df: pd.DataFrame,
    target_idx: int,
    n_neighbors: int = 5,
    distance_features: Sequence[str] | None = None,
    filter_columns: Sequence[str] | None = None,
    recent_first: bool = True,
) -> pd.DataFrame:
    """対象物件に類似する過去取引を抽出する.

    Args:
        df: 全取引データ。``run_pipeline()`` の出力を想定（categorical 情報が
            保持されており、``取引年`` / ``取引四半期`` を含む）。
        target_idx: 対象物件の ``df`` 上での index 値。
        n_neighbors: 返す類似物件数。母集団がこれ未満の場合は全件を返す。
        distance_features: 距離計算に使う連続特徴量。``None`` の場合は
            ``面積（㎡）`` / ``築年数`` / ``最寄駅：距離（分）`` を使う。
        filter_columns: 完全一致フィルタに使う列。``None`` の場合は
            ``種類`` / ``市区町村コード`` を使う。
        recent_first: True なら最終結果を取引時期の新しい順で並び替える。
            False なら距離の近い順のまま返す。

    Returns:
        類似物件の DataFrame。対象物件自身は除外される。
        新たに ``similarity_distance`` 列が追加される（正規化ユークリッド距離）。

    Raises:
        KeyError: ``df`` にフィルタ列や距離計算列が含まれていない場合。
        ValueError: ``target_idx`` が ``df`` に存在しない、または
            フィルタ後の母集団が（対象自身を除いて）空の場合。
    """
    filters = tuple(filter_columns) if filter_columns is not None else _DEFAULT_FILTER_COLUMNS
    features = (
        tuple(distance_features) if distance_features is not None else _DEFAULT_DISTANCE_FEATURES
    )

    if target_idx not in df.index:
        raise ValueError(f"対象 index が DataFrame に存在しません: target_idx={target_idx}")

    _ensure_columns(df, filters, "filter_columns")
    _ensure_columns(df, features, "distance_features")
    if recent_first:
        _ensure_columns(df, _RECENCY_COLUMNS, "recent_first=True")

    target_row = df.loc[target_idx]
    pool = _build_pool(df, target_row, filters)
    pool = pool.drop(index=target_idx, errors="ignore")

    if pool.empty:
        raise ValueError(
            f"類似物件の母集団が空です（target_idx={target_idx}, filters={list(filters)}）。"
            "フィルタを緩めるか、データ量を確認してください。"
        )

    distances = _compute_distance(pool, target_row, features)
    # 距離計算に欠損があった行を除外
    valid_mask = distances.notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info(
            "類似物件検索: 距離計算列に欠損があった %d 件を母集団から除外しました", n_dropped
        )
    distances = distances[valid_mask]

    if distances.empty:
        raise ValueError(
            f"距離計算可能な類似物件がありません（target_idx={target_idx}）。"
            "距離計算列の欠損状況を確認してください。"
        )

    if n_neighbors > len(distances):
        logger.warning(
            "類似物件検索: 要求件数 %d に対し母集団が %d 件しかありません。全件を返します。",
            n_neighbors,
            len(distances),
        )
        n_take = len(distances)
    else:
        n_take = n_neighbors

    top_indices = distances.nsmallest(n_take).index
    result = df.loc[top_indices].copy()
    result["similarity_distance"] = distances.loc[top_indices]

    if recent_first:
        # 取引年・取引四半期で降順（直近優先）
        result = result.sort_values(list(_RECENCY_COLUMNS), ascending=[False, False], kind="stable")

    return result
