"""高乖離物件 CSV 出力スクリプト（src.visualization.export_high_divergence）のテスト."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.visualization.export_high_divergence import (
    extract_model_features,
    load_features,
    order_output_columns,
)


def test_extract_model_features_returns_feature_names() -> None:
    """モデルの特徴量名がそのまま取り出されること."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = pd.Series(x.sum(axis=1))
    model = LGBMRegressor(n_estimators=5, verbose=-1).fit(x, y)

    features = extract_model_features({0.5: model})

    assert features == ["a", "b", "c"]


def test_order_output_columns_orders_id_then_metrics_then_trailing() -> None:
    """識別列 → 乖離指標 → 補助 の順に並び替えられること."""
    # 入力はわざとバラバラな列順にする
    table = pd.DataFrame(
        {
            "flag": ["要人手査定"],
            "ratio": [1.5],
            "市区町村コード": [13101],
            "index": [0],
            "divergence": [0.5],
            "quantile_median": [150],
            "empirical_median": [100],
        }
    )

    result = order_output_columns(table, id_columns=["市区町村コード"])

    # テーブルに存在する列だけが、規定順で並ぶ
    assert list(result.columns) == [
        "index",
        "市区町村コード",
        "divergence",
        "ratio",
        "quantile_median",
        "empirical_median",
        "flag",
    ]


def test_order_output_columns_skips_missing_columns() -> None:
    """テーブルに無い列はスキップされ、エラーにならないこと."""
    table = pd.DataFrame({"index": [0], "divergence": [0.4]})

    result = order_output_columns(table, id_columns=["市区町村コード"])

    assert list(result.columns) == ["index", "divergence"]


def test_load_features_raises_when_missing(tmp_path: Path) -> None:
    """特徴量ファイルが無い場合に FileNotFoundError."""
    missing = tmp_path / "features.csv"

    with pytest.raises(FileNotFoundError, match="特徴量ファイル"):
        load_features(missing)
