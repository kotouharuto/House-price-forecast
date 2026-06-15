"""分位点回帰学習モジュール（src.modeling.train_quantile）のテスト."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.modeling.train_quantile import (
    build_quantile_params,
    evaluate_coverage,
    save_quantile_models,
    train_quantile_models,
)


def _make_synthetic_dataset(
    n_samples: int = 200, n_features: int = 3, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """テスト用の合成データを生成する（学習が現実的な時間で終わる規模）."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # 目的変数: 特徴量の線形和 + ノイズ
    y = pd.Series(x.sum(axis=1) + rng.normal(scale=0.5, size=n_samples), name="target")
    return x, y


def _default_lgbm_params() -> dict:
    """テスト用の軽量 LightGBM パラメータ."""
    return {
        "n_estimators": 20,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 5,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 1,
    }


def _default_overrides() -> dict:
    """テスト用の quantile 上書きパラメータ."""
    return {"objective": "quantile", "metric": "quantile"}


def test_build_quantile_params_merges_lgbm_overrides_and_alpha() -> None:
    """lgbm + overrides + alpha がマージされること."""
    lgbm = {"n_estimators": 100, "objective": "huber", "metric": "rmse"}
    overrides = {"objective": "quantile", "metric": "quantile"}

    params = build_quantile_params(lgbm, overrides, alpha=0.1)

    # overrides が lgbm を上書き
    assert params["objective"] == "quantile"
    assert params["metric"] == "quantile"
    # lgbm の他項目は維持
    assert params["n_estimators"] == 100
    # alpha が追加
    assert params["alpha"] == 0.1


def test_train_quantile_models_returns_model_per_alpha() -> None:
    """alphas で指定した数だけモデルが返ること."""
    x, y = _make_synthetic_dataset()

    models = train_quantile_models(
        x,
        y,
        lgbm_params=_default_lgbm_params(),
        overrides=_default_overrides(),
        alphas=[0.1, 0.5, 0.9],
    )

    assert set(models.keys()) == {0.1, 0.5, 0.9}
    for alpha, model in models.items():
        assert isinstance(model, LGBMRegressor)
        # 各モデルが正しい alpha を持っていること
        assert model.alpha == alpha


def test_train_quantile_models_predictions_ordered_by_alpha() -> None:
    """α が大きいほど予測値が高い傾向にあること（中央値ベースで検証）."""
    x, y = _make_synthetic_dataset(n_samples=500)

    models = train_quantile_models(
        x,
        y,
        lgbm_params=_default_lgbm_params(),
        overrides=_default_overrides(),
        alphas=[0.1, 0.5, 0.9],
    )

    # 各 α での予測の中央値を比較（個別サンプルでは逆転もあり得るが、中央値は単調）
    median_preds = {alpha: float(np.median(m.predict(x))) for alpha, m in models.items()}
    assert median_preds[0.1] < median_preds[0.5] < median_preds[0.9]


def test_save_quantile_models_writes_three_files(tmp_path: Path) -> None:
    """3 モデルを low / med / high のファイル名で保存できること."""
    x, y = _make_synthetic_dataset()
    models = train_quantile_models(
        x,
        y,
        lgbm_params=_default_lgbm_params(),
        overrides=_default_overrides(),
        alphas=[0.1, 0.5, 0.9],
    )

    saved = save_quantile_models(models, model_dir=tmp_path)

    assert (tmp_path / "lgbm_quantile_low.pkl").exists()
    assert (tmp_path / "lgbm_quantile_med.pkl").exists()
    assert (tmp_path / "lgbm_quantile_high.pkl").exists()
    # 戻り値は α → パス の辞書
    assert set(saved.keys()) == {0.1, 0.5, 0.9}


def test_save_quantile_models_raises_when_count_is_wrong(tmp_path: Path) -> None:
    """モデル数が 3 でない場合に ValueError."""
    x, y = _make_synthetic_dataset()
    models = train_quantile_models(
        x,
        y,
        lgbm_params=_default_lgbm_params(),
        overrides=_default_overrides(),
        alphas=[0.1, 0.9],
    )

    with pytest.raises(ValueError, match="3 つの α"):
        save_quantile_models(models, model_dir=tmp_path)


def test_evaluate_coverage_returns_expected_keys() -> None:
    """カバレッジ評価が必要なキーを返すこと."""
    x, y = _make_synthetic_dataset(n_samples=300)
    models = train_quantile_models(
        x,
        y,
        lgbm_params=_default_lgbm_params(),
        overrides=_default_overrides(),
        alphas=[0.1, 0.5, 0.9],
    )

    result = evaluate_coverage(y, models, x)

    assert set(result.keys()) == {
        "nominal_coverage",
        "coverage_rate",
        "width_median_yen",
        "width_mean_yen",
    }
    # 名目カバレッジは upper - lower = 0.8
    assert result["nominal_coverage"] == pytest.approx(0.8)
    # 実カバレッジは 0〜1 の範囲
    assert 0.0 <= result["coverage_rate"] <= 1.0
    # 区間幅は非負
    assert result["width_median_yen"] >= 0
    assert result["width_mean_yen"] >= 0
