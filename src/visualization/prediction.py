"""予測ユーティリティ（点推定 + 予測区間）モジュール.

説明可能性ロードマップ Phase 2 の中核 API。Quantile Regression で学習した
3 つのモデル（low / median / high）から、対象物件の予測区間を返す。

不動産 AVM における「点推定 + 区間」のセット提示は、業務リスク管理
（融資判断・自動承認の閾値）や顧客説明（誠実な不確実性表示）の基盤となる。
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.utils.logger import get_logger

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
