"""予測ユーティリティ（点推定 + 予測区間）モジュール.

説明可能性ロードマップ Phase 2 の中核 API。Quantile Regression で学習した
3 つのモデル（low / median / high）から、対象物件の予測区間を返す。

不動産 AVM における「点推定 + 区間」のセット提示は、業務リスク管理
（融資判断・自動承認の閾値）や顧客説明（誠実な不確実性表示）の基盤となる。
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.utils.logger import get_logger
from src.visualization.similar_properties import find_similar_properties

logger = get_logger(__name__)

# 予測区間の DataFrame 列名（顧客向け UI からも参照される想定で定数化）
LOWER_COL = "lower"
MEDIAN_COL = "median"
UPPER_COL = "upper"

# 想定する分位点数（low / median / high の 3 つ）
_EXPECTED_NUM_QUANTILES = 3


def predict_with_interval(
    models: dict[float, LGBMRegressor],
    x: pd.DataFrame,
    return_yen: bool = True,
) -> pd.DataFrame:
    """分位点回帰モデル群を使って予測区間付きの予測値を返す.

    α を昇順に並べ、最小・中央・最大の 3 つを ``lower`` / ``median`` /
    ``upper`` 列にマッピングする。

    Args:
        models: ``{alpha: 学習済み LGBMRegressor}`` の辞書。3 件必要。
        x: 予測対象の特徴量 DataFrame。
        return_yen: True なら log 予測を ``np.exp`` で円換算する。
            False なら log スケールのまま返す。

    Returns:
        ``index = x.index``、``columns = [lower, median, upper]`` の DataFrame。

    Raises:
        ValueError: ``models`` の件数が 3 でない場合。
    """
    sorted_alphas = sorted(models.keys())
    if len(sorted_alphas) != _EXPECTED_NUM_QUANTILES:
        raise ValueError(
            f"3 つの α が必要です（low / median / high）: "
            f"got {len(sorted_alphas)} 件 {sorted_alphas}"
        )

    lower_alpha, median_alpha, upper_alpha = sorted_alphas
    lower = np.asarray(models[lower_alpha].predict(x))
    median = np.asarray(models[median_alpha].predict(x))
    upper = np.asarray(models[upper_alpha].predict(x))

    if return_yen:
        lower = np.exp(lower)
        median = np.exp(median)
        upper = np.exp(upper)

    return pd.DataFrame(
        {LOWER_COL: lower, MEDIAN_COL: median, UPPER_COL: upper},
        index=x.index,
    )


def load_quantile_models(model_dir: Path) -> dict[float, LGBMRegressor]:
    """``models/lgbm_quantile_{low,med,high}.pkl`` を一括ロードする.

    Args:
        model_dir: モデルディレクトリ。

    Returns:
        ``{alpha: モデル}`` の辞書。α はモデルの ``alpha`` 属性から復元する。

    Raises:
        FileNotFoundError: いずれかのモデルファイルが存在しない場合。
    """
    suffixes = ("low", "med", "high")
    models: dict[float, LGBMRegressor] = {}
    for suffix in suffixes:
        path = model_dir / f"lgbm_quantile_{suffix}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"分位点モデルが見つかりません: {path}")
        model: LGBMRegressor = joblib.load(path)
        alpha = float(model.alpha)
        models[alpha] = model
        logger.info(f"モデル読込: {path} (α={alpha})")
    return models


def empirical_interval_from_similar(
    df: pd.DataFrame,
    target_idx: int,
    n_similar: int = 30,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    price_column: str = "取引価格（総額）",
) -> dict[str, float]:
    """類似物件の実取引価格の分位点から予測区間を出す.

    Returns:
        {lower, median, upper, n_used} の辞書
    """
    # Phase 1 の関数で類似 N 件を取得
    similar = find_similar_properties(df, target_idx, n_neighbors=n_similar)
    prices = similar[price_column].astype(float)

    return {
        "lower": float(prices.quantile(lower_q)),
        "median": float(prices.quantile(0.50)),
        "upper": float(prices.quantile(upper_q)),
        "n_used": len(prices),
    }


def compare_intervals(
    quantile_interval: dict[str, float],  # Phase 2の出力
    empirical_interval: dict[str, float],  # Phase 3の出力
    divergence_threshold: float = 0.30,  # 中央値が30%以上ずれたら警告
) -> dict[str, Any]:
    """2 つの区間を比較し、乖離度と信頼度フラグを返す.

    Returns:
        {
          divergence: float,        # 中央値の相対乖離
          is_reliable: bool,        # 乖離が閾値以内か
          flag: str,                # "高信頼" / "要人手査定"
          ...
        }
    """
    q_med = quantile_interval[MEDIAN_COL]
    e_med = empirical_interval[MEDIAN_COL]

    # ゼロ除算ガード（実証中央値が 0 になることは実質ないが念のため）
    if e_med <= 0:
        raise ValueError(f"実証中央値が不正です: {e_med}")

    # 中央値の相対乖離（実証側を基準にする = 事実を分母に置く）
    divergence = abs(q_med - e_med) / e_med
    is_reliable = divergence <= divergence_threshold

    # 区間の重なり（overlap）も見ると、より精密な判定ができる
    overlap_lower = max(quantile_interval[LOWER_COL], empirical_interval[LOWER_COL])
    overlap_upper = min(quantile_interval[UPPER_COL], empirical_interval[UPPER_COL])
    has_overlap = overlap_lower <= overlap_upper

    return {
        "divergence": divergence,
        "is_reliable": is_reliable,
        "has_overlap": has_overlap,
        "flag": "高信頼" if is_reliable and has_overlap else "要人手査定",
        "quantile_median": q_med,
        "empirical_median": e_med,
        "n_similar_used": empirical_interval.get("n_used"),
    }
