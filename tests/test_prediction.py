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
    compare_intervals,
    empirical_interval_from_similar,
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


# ============================================================
# Phase 3: 実証的区間（empirical_interval_from_similar / compare_intervals）
# ============================================================


def _make_similar_df(prices: list[int]) -> pd.DataFrame:
    """実証的区間テスト用の取引データを生成する.

    対象物件（idx=0）と同じ種類・市区町村・距離特徴を持つ候補を ``prices`` の
    件数だけ並べる。距離特徴を全件同一にすることで、``find_similar_properties``
    が全候補を等距離で選び、価格分位点が決定的に計算できるようにする。

    Args:
        prices: 候補物件（idx=1 以降）の取引価格リスト。

    Returns:
        idx=0 が対象、idx>=1 が候補となる DataFrame。
    """
    n_candidates = len(prices)
    n_rows = n_candidates + 1  # 対象 1 件 + 候補
    # 対象の価格は分位点計算に影響しないようダミー値を置く
    all_prices = [99_999_999, *prices]
    return pd.DataFrame(
        {
            "種類": ["中古マンション等"] * n_rows,
            "市区町村コード": [13112] * n_rows,
            "面積（㎡）": [60.0] * n_rows,
            "築年数": [20.0] * n_rows,
            "最寄駅：距離（分）": [5.0] * n_rows,
            "取引年": [2024] * n_rows,
            "取引四半期": [1] * n_rows,
            "取引価格（総額）": all_prices,
        }
    )


def test_empirical_interval_basic_ordering() -> None:
    """lower < median < upper の順序関係が成り立つこと."""
    prices = list(range(30_000_000, 90_000_000, 2_000_000))  # 30 件
    df = _make_similar_df(prices)

    result = empirical_interval_from_similar(df, target_idx=0, n_similar=30)

    assert result["lower"] < result["median"] < result["upper"]
    assert result["n_used"] == len(prices)


def test_empirical_interval_quantile_values_match_pandas() -> None:
    """返す分位点が pandas の quantile 計算と一致すること."""
    prices = [30_000_000, 40_000_000, 50_000_000, 60_000_000, 70_000_000]
    df = _make_similar_df(prices)

    result = empirical_interval_from_similar(
        df, target_idx=0, n_similar=30, lower_q=0.10, upper_q=0.90
    )

    expected = pd.Series(prices, dtype=float)
    assert result["lower"] == pytest.approx(expected.quantile(0.10))
    assert result["median"] == pytest.approx(expected.quantile(0.50))
    assert result["upper"] == pytest.approx(expected.quantile(0.90))


def test_empirical_interval_n_used_reflects_actual_count() -> None:
    """候補が要求件数より少ないとき n_used が実数を反映すること."""
    prices = [40_000_000, 50_000_000, 60_000_000]  # 3 件しかない
    df = _make_similar_df(prices)

    result = empirical_interval_from_similar(df, target_idx=0, n_similar=30)

    assert result["n_used"] == 3


def test_empirical_interval_excludes_target_price() -> None:
    """対象自身の価格（ダミー値）が分位点に混入しないこと."""
    prices = [40_000_000, 50_000_000, 60_000_000]
    df = _make_similar_df(prices)  # 対象は 99,999,999 円

    result = empirical_interval_from_similar(df, target_idx=0, n_similar=30)

    # 対象のダミー価格が混入していれば upper は 99,999,999 に近づくはず
    assert result["upper"] <= 60_000_000


def test_compare_intervals_reliable_when_close() -> None:
    """2 区間が近いとき is_reliable=True, flag='高信頼'."""
    quantile_interval = {LOWER_COL: 4_700, MEDIAN_COL: 5_200, UPPER_COL: 5_800}
    empirical_interval = {
        LOWER_COL: 4_900,
        MEDIAN_COL: 5_500,
        UPPER_COL: 6_200,
        "n_used": 30,
    }

    result = compare_intervals(quantile_interval, empirical_interval)

    assert result["is_reliable"] is True
    assert result["has_overlap"] is True
    assert result["flag"] == "高信頼"
    assert result["n_similar_used"] == 30


def test_compare_intervals_flags_manual_when_divergent() -> None:
    """中央値が大きく乖離し区間も重ならないとき '要人手査定'."""
    quantile_interval = {LOWER_COL: 4_700, MEDIAN_COL: 5_200, UPPER_COL: 5_800}
    empirical_interval = {
        LOWER_COL: 6_500,
        MEDIAN_COL: 8_000,
        UPPER_COL: 9_000,
        "n_used": 30,
    }

    result = compare_intervals(quantile_interval, empirical_interval)

    assert result["is_reliable"] is False
    assert result["has_overlap"] is False
    assert result["flag"] == "要人手査定"


def test_compare_intervals_divergence_value() -> None:
    """divergence が実証中央値基準の相対乖離になっていること."""
    quantile_interval = {LOWER_COL: 4_000, MEDIAN_COL: 5_500, UPPER_COL: 7_000}
    empirical_interval = {
        LOWER_COL: 4_000,
        MEDIAN_COL: 5_000,
        UPPER_COL: 6_000,
        "n_used": 20,
    }

    result = compare_intervals(quantile_interval, empirical_interval)

    # |5500 - 5000| / 5000 = 0.10
    assert result["divergence"] == pytest.approx(0.10)


def test_compare_intervals_raises_on_nonpositive_empirical_median() -> None:
    """実証中央値が 0 以下のとき ValueError."""
    quantile_interval = {LOWER_COL: 4_700, MEDIAN_COL: 5_200, UPPER_COL: 5_800}
    empirical_interval = {LOWER_COL: 0, MEDIAN_COL: 0, UPPER_COL: 0, "n_used": 0}

    with pytest.raises(ValueError, match="実証中央値"):
        compare_intervals(quantile_interval, empirical_interval)
