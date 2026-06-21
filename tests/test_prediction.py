"""予測ユーティリティ（src.visualization.prediction）のテスト."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from src.visualization.prediction import (
    COUNT_COL,
    DIVERGENCE_COL,
    LOWER_COL,
    MEDIAN_COL,
    NAME_COL,
    OVER_RATE_COL,
    RATIO_COL,
    RATIO_MEDIAN_COL,
    UPPER_COL,
    add_interpretability_columns,
    aggregate_divergence_by_municipality,
    build_divergence_table,
    compare_intervals,
    divergence_band,
    divergence_direction,
    empirical_interval_from_similar,
    filter_high_divergence,
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


# ============================================================
# 乖離度テーブル（build_divergence_table / filter_high_divergence）
# ============================================================


def _make_divergence_df(n_rows: int = 8, seed: int = 1) -> pd.DataFrame:
    """乖離度テーブルテスト用の DataFrame.

    全行が同一の種類・市区町村コードを持つため、各行は他の全行を類似候補とする。
    モデル特徴量 (a/b/c) と類似物件検索に必要な列を併せ持つ。
    """
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(rng.normal(size=(n_rows, 3)), columns=["a", "b", "c"])
    return features.assign(
        種類=["中古マンション等"] * n_rows,
        市区町村コード=[13112] * n_rows,
        面積=np.linspace(50.0, 80.0, n_rows),
        築年数=np.linspace(5.0, 30.0, n_rows),
        最寄駅_距離=np.linspace(3.0, 12.0, n_rows),
        取引年=[2024] * n_rows,
        取引四半期=[1] * n_rows,
        取引価格=np.linspace(30_000_000, 80_000_000, n_rows),
    ).rename(
        columns={
            "面積": "面積（㎡）",
            "最寄駅_距離": "最寄駅：距離（分）",
            "取引価格": "取引価格（総額）",
        }
    )


def test_build_divergence_table_has_expected_columns() -> None:
    """テーブルに識別列と乖離度・フラグ列が揃っていること."""
    models = _train_tiny_quantile_models()
    df = _make_divergence_df()

    table = build_divergence_table(
        df,
        models,
        model_features=["a", "b", "c"],
        n_similar=5,
        id_columns=["市区町村コード", "面積（㎡）"],
    )

    assert len(table) == len(df)
    for col in (
        "index",
        "市区町村コード",
        "面積（㎡）",
        DIVERGENCE_COL,
        RATIO_COL,
        "signed_pct",
        "direction",
        "band",
        "quantile_median",
        "empirical_median",
        "n_used",
        "flag",
    ):
        assert col in table.columns
    # 乖離度は非負、フラグは 2 値のいずれか
    assert (table[DIVERGENCE_COL] >= 0).all()
    assert table["flag"].isin(["高信頼", "要人手査定"]).all()
    # ratio と divergence は |ratio - 1| == divergence の関係
    np.testing.assert_allclose(
        (table[RATIO_COL] - 1.0).abs().to_numpy(),
        table[DIVERGENCE_COL].to_numpy(),
        rtol=1e-9,
    )


def test_filter_high_divergence_sorts_descending_and_filters() -> None:
    """閾値超の行のみを乖離度降順で返すこと."""
    table = pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            DIVERGENCE_COL: [0.05, 0.45, 0.30, 0.60],
        }
    )

    result = filter_high_divergence(table, threshold=0.30)

    # 0.30 は「超える」ではないので除外、0.45 と 0.60 が残る
    assert result["index"].tolist() == [3, 1]
    assert result[DIVERGENCE_COL].tolist() == [0.60, 0.45]


def test_filter_high_divergence_empty_when_all_below() -> None:
    """全行が閾値以下なら空の DataFrame を返すこと."""
    table = pd.DataFrame({"index": [0, 1], DIVERGENCE_COL: [0.10, 0.20]})

    result = filter_high_divergence(table, threshold=0.30)

    assert result.empty


# ============================================================
# 解釈用列（ratio / signed_pct / direction / band）
# ============================================================


def test_divergence_direction_labels() -> None:
    """ratio から高値 / 安値 / 一致が正しく判定されること."""
    assert divergence_direction(1.30) == "高値"
    assert divergence_direction(0.70) == "安値"
    assert divergence_direction(1.00) == "一致"


def test_divergence_band_is_symmetric_for_high_and_low() -> None:
    """高値側・安値側で対称にバンド判定されること（fold ベース）."""
    # 2倍 と 半額 はどちらも fold=2.0 → 重度
    assert divergence_band(2.0) == "重度"
    assert divergence_band(0.5) == "重度"
    # 1.3倍未満 は軽度、1.3〜2.0倍 は中度
    assert divergence_band(1.2) == "軽度"
    assert divergence_band(1.5) == "中度"
    # 対称: 1/1.5 ≒ 0.667 も中度
    assert divergence_band(1.0 / 1.5) == "中度"


def test_divergence_band_raises_on_nonpositive_ratio() -> None:
    """ratio が 0 以下なら ValueError."""
    with pytest.raises(ValueError, match="ratio"):
        divergence_band(0.0)


def test_add_interpretability_columns_values() -> None:
    """ratio / signed_pct / direction / band が中央値から正しく算出されること."""
    table = pd.DataFrame(
        {
            "quantile_median": [12_000_000, 3_500_000],
            "empirical_median": [10_000_000, 7_000_000],
        }
    )

    result = add_interpretability_columns(table)

    # 1行目: 1200万 / 1000万 = 1.2倍, +20%, 高値, 軽度
    assert result.loc[0, RATIO_COL] == pytest.approx(1.20)
    assert result.loc[0, "signed_pct"] == pytest.approx(20.0)
    assert result.loc[0, "direction"] == "高値"
    assert result.loc[0, "band"] == "軽度"
    # 2行目: 350万 / 700万 = 0.5倍, -50%, 安値, 重度
    assert result.loc[1, RATIO_COL] == pytest.approx(0.50)
    assert result.loc[1, "signed_pct"] == pytest.approx(-50.0)
    assert result.loc[1, "direction"] == "安値"
    assert result.loc[1, "band"] == "重度"


def test_add_interpretability_columns_handles_empty() -> None:
    """空テーブルでも列を付与して返すこと."""
    table = pd.DataFrame({"quantile_median": [], "empirical_median": []})

    result = add_interpretability_columns(table)

    for col in (RATIO_COL, "signed_pct", "direction", "band"):
        assert col in result.columns
    assert result.empty


# ============================================================
# 市区町村別の乖離集計（aggregate_divergence_by_municipality）
# ============================================================


def _make_divergence_table() -> pd.DataFrame:
    """市区町村集計テスト用の乖離度テーブルを生成する.

    13101: ratio が高め（モデル高値傾向） / 13109: ratio が低め（モデル安値傾向） /
    13999: 件数が少ない（min_count フィルタ確認用）。
    """
    return pd.DataFrame(
        {
            "市区町村コード": [13101, 13101, 13101, 13109, 13109, 13109, 13999],
            RATIO_COL: [1.5, 1.4, 1.6, 0.7, 0.8, 0.6, 2.0],
        }
    )


def test_aggregate_high_mode_sorts_by_ratio_desc() -> None:
    """mode='high' で ratio 中央値の降順に並ぶこと."""
    table = _make_divergence_table()

    result = aggregate_divergence_by_municipality(table, mode="high", min_count=3)

    # 13101（中央値 1.5）が 13109（中央値 0.7）より上位
    assert list(result["市区町村コード"]) == [13101, 13109]
    assert result[RATIO_MEDIAN_COL].is_monotonic_decreasing


def test_aggregate_low_mode_sorts_by_ratio_asc() -> None:
    """mode='low' で ratio 中央値の昇順に並ぶこと."""
    table = _make_divergence_table()

    result = aggregate_divergence_by_municipality(table, mode="low", min_count=3)

    # 安値傾向の 13109 が上位
    assert list(result["市区町村コード"]) == [13109, 13101]
    assert result[RATIO_MEDIAN_COL].is_monotonic_increasing


def test_aggregate_respects_min_count() -> None:
    """min_count 未満の市区町村が除外されること."""
    table = _make_divergence_table()

    result = aggregate_divergence_by_municipality(table, min_count=3)

    # 13999 は 1 件なので除外される
    assert 13999 not in set(result["市区町村コード"])
    assert set(result["市区町村コード"]) == {13101, 13109}


def test_aggregate_columns_and_over_rate() -> None:
    """集計列が揃い、over_rate が ratio>1 の割合になっていること."""
    table = _make_divergence_table()

    result = aggregate_divergence_by_municipality(table, min_count=3)

    for col in (COUNT_COL, RATIO_MEDIAN_COL, OVER_RATE_COL):
        assert col in result.columns
    # 13101 は 3 件すべて ratio>1 → over_rate 100%、13109 は 0%
    row_13101 = result[result["市区町村コード"] == 13101].iloc[0]
    row_13109 = result[result["市区町村コード"] == 13109].iloc[0]
    assert row_13101[COUNT_COL] == 3
    assert row_13101[OVER_RATE_COL] == pytest.approx(100.0)
    assert row_13109[OVER_RATE_COL] == pytest.approx(0.0)


def test_aggregate_adds_name_when_dict_given() -> None:
    """name_by_code を渡すと市区町村名列が付与されること."""
    table = _make_divergence_table()
    name_by_code = {13101: "千代田区", 13109: "品川区"}

    result = aggregate_divergence_by_municipality(table, min_count=3, name_by_code=name_by_code)

    assert NAME_COL in result.columns
    assert set(result[NAME_COL]) == {"千代田区", "品川区"}


def test_aggregate_unknown_code_falls_back_to_code_string() -> None:
    """辞書に無いコードは文字列化したコードで埋められること."""
    table = _make_divergence_table()
    name_by_code = {13101: "千代田区"}  # 13109 は欠落

    result = aggregate_divergence_by_municipality(table, min_count=3, name_by_code=name_by_code)

    name_13109 = result[result["市区町村コード"] == 13109][NAME_COL].iloc[0]
    assert name_13109 == "13109"


def test_aggregate_raises_on_invalid_mode() -> None:
    """mode が high/low 以外で ValueError."""
    table = _make_divergence_table()

    with pytest.raises(ValueError, match="mode"):
        aggregate_divergence_by_municipality(table, mode="middle")


def test_aggregate_raises_when_required_column_missing() -> None:
    """必須列が無いとき KeyError."""
    table = pd.DataFrame({"市区町村コード": [13101, 13101]})  # ratio が無い

    with pytest.raises(KeyError, match=RATIO_COL):
        aggregate_divergence_by_municipality(table)
