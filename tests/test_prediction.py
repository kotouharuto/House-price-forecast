"""予測ユーティリティ（src.visualization.prediction）のテスト."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.visualization.prediction import (
    LOWER_COL,
    MEDIAN_COL,
    UPPER_COL,
    load_quantile_models,
    predict_with_interval,
)


def _train_tiny_quantile_models(seed: int = 42) -> dict[float, LGBMRegressor]:
    """テスト用の小規模分位点モデルを学習する."""
    rng = np.random.default_rng(seed)
    x_train = pd.DataFrame(
        rng.normal(size=(200, 3)),
        columns=["a", "b", "c"],
    )
    y_train = pd.Series(x_train.sum(axis=1) + rng.normal(scale=0.3, size=200))

    base_params = {
        "n_estimators": 20,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 5,
        "objective": "quantile",
        "metric": "quantile",
        "random_state": seed,
        "verbose": -1,
        "n_jobs": 1,
    }
    return {
        alpha: LGBMRegressor(**base_params, alpha=alpha).fit(x_train, y_train)
        for alpha in (0.1, 0.5, 0.9)
    }


def _make_x_test() -> pd.DataFrame:
    """テスト用の予測対象 DataFrame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])


def test_predict_with_interval_returns_three_columns() -> None:
    """戻り値は lower / median / upper の 3 列であること."""
    models = _train_tiny_quantile_models()
    x_test = _make_x_test()

    result = predict_with_interval(models, x_test, return_yen=False)

    assert list(result.columns) == [LOWER_COL, MEDIAN_COL, UPPER_COL]
    assert len(result) == len(x_test)
    # x_test の index を引き継いでいること
    assert (result.index == x_test.index).all()


def test_predict_with_interval_median_between_bounds_on_average() -> None:
    """中央値予測の平均が lower と upper の間にあること."""
    models = _train_tiny_quantile_models()
    x_test = _make_x_test()

    result = predict_with_interval(models, x_test, return_yen=False)

    # 個別サンプルでは逆転もあり得るが、平均は単調になる
    assert result[LOWER_COL].mean() < result[MEDIAN_COL].mean() < result[UPPER_COL].mean()


def test_predict_with_interval_return_yen_applies_exp() -> None:
    """return_yen=True で np.exp が適用されること."""
    models = _train_tiny_quantile_models()
    x_test = _make_x_test()

    result_log = predict_with_interval(models, x_test, return_yen=False)
    result_yen = predict_with_interval(models, x_test, return_yen=True)

    # exp 変換後は元の値より大きく、かつ exp(log) = yen の関係
    np.testing.assert_allclose(
        result_yen[MEDIAN_COL].to_numpy(),
        np.exp(result_log[MEDIAN_COL].to_numpy()),
        rtol=1e-9,
    )


def test_predict_with_interval_raises_when_count_is_wrong() -> None:
    """モデル数が 3 でない場合に ValueError."""
    models = _train_tiny_quantile_models()
    # 1 つ削って 2 件にする
    models.pop(0.5)
    x_test = _make_x_test()

    with pytest.raises(ValueError, match="3 つの α"):
        predict_with_interval(models, x_test)


def test_load_quantile_models_round_trip(tmp_path: Path) -> None:
    """保存したモデルを読み込めること（α 属性から辞書が再構成される）."""
    models = _train_tiny_quantile_models()
    # 手動で 3 ファイルを保存（save_quantile_models を介さない経路もテスト）
    name_by_alpha = {0.1: "low", 0.5: "med", 0.9: "high"}
    for alpha, name in name_by_alpha.items():
        joblib.dump(models[alpha], tmp_path / f"lgbm_quantile_{name}.pkl")

    loaded = load_quantile_models(tmp_path)

    assert set(loaded.keys()) == {0.1, 0.5, 0.9}
    for alpha, model in loaded.items():
        assert isinstance(model, LGBMRegressor)
        assert model.alpha == alpha


def test_load_quantile_models_raises_when_file_missing(tmp_path: Path) -> None:
    """ファイルが無い場合に FileNotFoundError."""
    # low だけ保存
    models = _train_tiny_quantile_models()
    joblib.dump(models[0.1], tmp_path / "lgbm_quantile_low.pkl")

    with pytest.raises(FileNotFoundError, match="lgbm_quantile_med.pkl"):
        load_quantile_models(tmp_path)
