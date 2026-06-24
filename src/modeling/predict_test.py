"""テストセットの予測結果 CSV を生成するスクリプト（点予測 + 予測区間）.

説明可能性ロードマップ Phase 5 の起点。``outputs/test_predictions.csv`` を、
現行データ・現行モデルで再現可能に作り直す。点予測（``lgbm_model.pkl``）に加えて
分位点回帰モデルの予測区間（``pred_lower_yen`` / ``pred_upper_yen``）を付与し、
PICP / PIAW などの区間評価を後続フェーズで計算できるようにする。

実行例::

    uv run python -m src.modeling.predict_test

学習（``train.py``）と同じ前処理・同じ train/test 分割を再現するため、
点予測モデルが学習に使ったのと同じ held-out テストセットに対して評価できる。
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.modeling.train import prepare_dataset_with_frame  # noqa: E402
from src.utils.config import load_model_params  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.visualization.aggregate import (  # noqa: E402
    ACTUAL_LOG_COL,
    ACTUAL_PRICE_COL,
    APE_COL,
    ERROR_RATE_COL,
    ERROR_YEN_COL,
    PRED_LOG_COL,
    PRED_PRICE_COL,
    PRICE_BAND_COL,
)
from src.visualization.prediction import (  # noqa: E402
    LOWER_COL,
    MEDIAN_COL,
    UPPER_COL,
    load_quantile_models,
    predict_with_interval,
)

logger = get_logger(__name__)

_MODEL_DIR = _PROJECT_ROOT / "models"
_POINT_MODEL_PATH = _MODEL_DIR / "lgbm_model.pkl"
_OUTPUT_PATH = _PROJECT_ROOT / "outputs" / "test_predictions.csv"

# 予測区間の出力列名（Phase 5 の評価関数が参照する）
PRED_LOWER_COL = "pred_lower_yen"
PRED_MEDIAN_COL = "pred_median_yen"
PRED_UPPER_COL = "pred_upper_yen"
INTERVAL_WIDTH_COL = "interval_width_yen"
ABS_ERROR_COL = "abs_error_yen"

# 価格帯のビン（円）とラベル。既存 CSV のラベル体系に合わせる。
# 区間は左閉右開 [下限, 上限) とする（例: 2000万~5000万 = 2000万以上5000万未満）。
_BAND_BINS = (0.0, 2e7, 5e7, 1e8, 3e8, np.inf)
_BAND_LABELS = ("~2000万", "2000万~5000万", "5000万~1億", "1億~3億", "3億~")


def assign_price_band(actual_yen: pd.Series) -> pd.Series:
    """実取引価格（円）を価格帯ラベルに割り当てる.

    Args:
        actual_yen: 実取引価格の Series（円）。

    Returns:
        価格帯ラベルの Series（文字列）。
    """
    band = pd.cut(
        actual_yen,
        bins=list(_BAND_BINS),
        labels=list(_BAND_LABELS),
        right=False,
    )
    return band.astype(str)


def build_test_predictions(
    test_df: pd.DataFrame,
    y_test_log: pd.Series,
    pred_log: np.ndarray,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    """テスト特徴量・実測・点予測・区間予測から出力用 DataFrame を組み立てる.

    Args:
        test_df: テスト行の全特徴量（``test_df.index`` がオリジナル index）。
        y_test_log: テスト目的変数（log スケール）。
        pred_log: 点予測（log スケール）。
        intervals: ``predict_with_interval`` の出力（円スケール、lower/median/upper）。

    Returns:
        評価・可視化に使う列を備えた DataFrame。
    """
    actual_yen = np.exp(y_test_log.to_numpy())
    pred_yen = np.exp(np.asarray(pred_log))
    error = pred_yen - actual_yen
    abs_error = np.abs(error)

    aligned = intervals.loc[test_df.index]

    out = test_df.copy()
    out.insert(0, "original_index", out.index)
    out[ACTUAL_LOG_COL] = y_test_log.to_numpy()
    out[PRED_LOG_COL] = pred_log
    out[ACTUAL_PRICE_COL] = actual_yen
    out[PRED_PRICE_COL] = pred_yen
    out[ERROR_YEN_COL] = error
    out[ABS_ERROR_COL] = abs_error
    out[ERROR_RATE_COL] = error / actual_yen * 100
    out[APE_COL] = abs_error / actual_yen * 100
    out[PRICE_BAND_COL] = assign_price_band(out[ACTUAL_PRICE_COL]).to_numpy()
    out[PRED_LOWER_COL] = aligned[LOWER_COL].to_numpy()
    out[PRED_MEDIAN_COL] = aligned[MEDIAN_COL].to_numpy()
    out[PRED_UPPER_COL] = aligned[UPPER_COL].to_numpy()
    out[INTERVAL_WIDTH_COL] = out[PRED_UPPER_COL] - out[PRED_LOWER_COL]

    return out.reset_index(drop=True)


def main() -> None:
    """エントリポイント: データ準備 → 分割 → 点/区間予測 → CSV 出力."""
    start_time = time.time()

    split_params = load_model_params()["split"]
    df, x, y = prepare_dataset_with_frame()
    _, x_test, _, y_test = train_test_split(x, y, **split_params)

    point_model = joblib.load(_POINT_MODEL_PATH)
    quantile_models = load_quantile_models(_MODEL_DIR)

    pred_log = point_model.predict(x_test)
    intervals = predict_with_interval(quantile_models, x_test)
    test_df = df.loc[x_test.index]

    out = build_test_predictions(test_df, y_test, pred_log, intervals)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(_OUTPUT_PATH, index=False)

    logger.info(
        f"テスト予測 CSV を出力: {len(out)} 件 → {_OUTPUT_PATH} "
        f"(区間幅 中央値 {out[INTERVAL_WIDTH_COL].median():,.0f} 円)"
    )
    logger.info(f"テスト予測生成完了: 所要時間 {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
