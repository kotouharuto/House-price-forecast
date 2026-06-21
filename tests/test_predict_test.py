"""テスト予測 CSV 生成（src.modeling.predict_test）のテスト.

区間予測の付与と派生列の組み立てロジック（純粋関数）を検証する。
モデル学習・前処理パイプラインを伴う ``main`` は対象外。
"""

import numpy as np
import pandas as pd
import pytest

from src.modeling.predict_test import (
    ABS_ERROR_COL,
    INTERVAL_WIDTH_COL,
    PRED_LOWER_COL,
    PRED_MEDIAN_COL,
    PRED_UPPER_COL,
    assign_price_band,
    build_test_predictions,
)
from src.visualization.aggregate import (
    ACTUAL_PRICE_COL,
    APE_COL,
    ERROR_YEN_COL,
    PRED_PRICE_COL,
    PRICE_BAND_COL,
)
from src.visualization.prediction import LOWER_COL, MEDIAN_COL, UPPER_COL


def test_assign_price_band_boundaries() -> None:
    """価格帯ラベルが境界（左閉右開）で正しく割り当てられること."""
    prices = pd.Series([1_000_000, 20_000_000, 49_999_999, 50_000_000, 150_000_000, 400_000_000])

    bands = assign_price_band(prices)

    assert bands.tolist() == [
        "~2000万",
        "2000万~5000万",  # 2000万ちょうどは下側の帯に入る（左閉）
        "2000万~5000万",
        "5000万~1億",
        "1億~3億",
        "3億~",
    ]


def _make_test_inputs() -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.DataFrame]:
    """build_test_predictions 用の入力を作る（index を非連番にして整合を確認）."""
    idx = [10, 25, 47]
    test_df = pd.DataFrame({"種類": ["A", "B", "C"], "面積（㎡）": [50, 60, 70]}, index=idx)
    # log 価格（exp で 3000万 / 5000万 / 1.2億 相当）
    y_log = pd.Series(np.log([30_000_000, 50_000_000, 120_000_000]), index=idx)
    pred_log = np.log([33_000_000, 48_000_000, 100_000_000])
    # 区間（円）。順序確認のため index をわざと入れ替えて渡す
    intervals = pd.DataFrame(
        {
            LOWER_COL: [25_000_000, 40_000_000, 90_000_000],
            MEDIAN_COL: [32_000_000, 49_000_000, 110_000_000],
            UPPER_COL: [40_000_000, 60_000_000, 140_000_000],
        },
        index=[47, 10, 25],
    )
    return test_df, y_log, pred_log, intervals


def test_build_test_predictions_columns_and_count() -> None:
    """出力に必要列が揃い、件数が一致すること."""
    test_df, y_log, pred_log, intervals = _make_test_inputs()

    out = build_test_predictions(test_df, y_log, pred_log, intervals)

    assert len(out) == len(test_df)
    for col in (
        "original_index",
        ACTUAL_PRICE_COL,
        PRED_PRICE_COL,
        ERROR_YEN_COL,
        ABS_ERROR_COL,
        APE_COL,
        PRICE_BAND_COL,
        PRED_LOWER_COL,
        PRED_MEDIAN_COL,
        PRED_UPPER_COL,
        INTERVAL_WIDTH_COL,
    ):
        assert col in out.columns


def test_build_test_predictions_derived_values() -> None:
    """点予測・誤差・区間幅が円スケールで正しく計算されること."""
    test_df, y_log, pred_log, intervals = _make_test_inputs()

    out = build_test_predictions(test_df, y_log, pred_log, intervals).set_index("original_index")

    # idx=10: 実3000万 / 予測3300万
    row = out.loc[10]
    assert row[ACTUAL_PRICE_COL] == pytest.approx(30_000_000, rel=1e-9)
    assert row[PRED_PRICE_COL] == pytest.approx(33_000_000, rel=1e-9)
    assert row[ERROR_YEN_COL] == pytest.approx(3_000_000, rel=1e-9)
    assert row[APE_COL] == pytest.approx(10.0, rel=1e-9)
    # 区間幅 = upper - lower
    assert row[INTERVAL_WIDTH_COL] == pytest.approx(row[PRED_UPPER_COL] - row[PRED_LOWER_COL])


def test_build_test_predictions_aligns_intervals_by_index() -> None:
    """区間が index で整合され、行の取り違えが起きないこと."""
    test_df, y_log, pred_log, intervals = _make_test_inputs()

    out = build_test_predictions(test_df, y_log, pred_log, intervals).set_index("original_index")

    # intervals は index で引き当てるので、順序が違っても idx=10 の lower は 40,000,000
    assert out.loc[10, PRED_LOWER_COL] == 40_000_000
    assert out.loc[25, PRED_LOWER_COL] == 90_000_000
    assert out.loc[47, PRED_LOWER_COL] == 25_000_000


def test_build_test_predictions_keeps_features() -> None:
    """元の特徴量列が保持されること."""
    test_df, y_log, pred_log, intervals = _make_test_inputs()

    out = build_test_predictions(test_df, y_log, pred_log, intervals)

    assert "種類" in out.columns
    assert "面積（㎡）" in out.columns
