"""予測結果集計モジュール（src.visualization.aggregate）のテスト."""

import math
from pathlib import Path

import pandas as pd
import pytest

from src.visualization.aggregate import (
    _DEFAULT_PRED_PATH,
    aggregate_by_station,
    aggregate_by_ward,
    aggregate_predictions,
    available_stations,
    default_predictions_path,
    filter_predictions,
    load_predictions,
    price_band_order,
    station_map_summary,
    summarize_metrics,
    ward_map_summary,
    worst_properties,
)


def _sample_df() -> pd.DataFrame:
    """集計テスト用の最小データフレームを生成する.

    市区町村コード 13123 が2件・13213 が1件、駅は A駅2件・B駅1件。
    """
    return pd.DataFrame(
        {
            "市区町村コード": [13123, 13123, 13213],
            "最寄駅：名称": ["A駅", "A駅", "B駅"],
            "最寄駅：緯度": [35.68, 35.70, 35.74],
            "最寄駅：経度": [139.86, 139.88, 139.47],
            "種類": ["中古マンション等", "中古マンション等", "宅地(建物)"],
            "actual_price_band": ["1000万~5000万", "1000万~5000万", "5000万~1億"],
            "面積（㎡）": [40.0, 60.0, 80.0],
            "築年数": [5.0, 15.0, 30.0],
            "山手線内側": [1, 1, 0],
            "pred_price_yen": [100, 300, 1000],
            "actual_price_yen": [110, 290, 900],
            "error_yen": [-10, 10, 100],  # pred - actual
            "ape_percent": [10.0, 20.0, 30.0],
        }
    )


def test_aggregate_predictions_computes_stats() -> None:
    """件数・平均・中央値・MAPE・Median APE が正しく計算されること."""
    result = aggregate_predictions(_sample_df(), "市区町村コード")
    row = result.set_index("市区町村コード").loc[13123]

    assert row["count"] == 2
    assert row["pred_price_mean"] == 200.0  # (100 + 300) / 2
    assert row["pred_price_median"] == 200.0
    assert row["actual_price_mean"] == 200.0  # (110 + 290) / 2
    assert row["mape"] == 15.0  # (10 + 20) / 2
    assert row["median_ape"] == 15.0


def test_aggregate_predictions_without_coords_has_no_latlon() -> None:
    """coord_cols 未指定なら lat/lon 列を持たないこと."""
    result = aggregate_predictions(_sample_df(), "市区町村コード")

    assert "lat" not in result.columns
    assert "lon" not in result.columns


def test_aggregate_by_ward_groups_by_code() -> None:
    """行政区集計が市区町村コード単位でまとまること."""
    result = aggregate_by_ward(_sample_df())

    assert set(result["市区町村コード"]) == {13123, 13213}
    assert len(result) == 2


def test_aggregate_by_station_adds_representative_coords() -> None:
    """駅集計が代表座標(lat/lon, グループ平均)を付与すること."""
    result = aggregate_by_station(_sample_df())
    row = result.set_index("最寄駅：名称").loc["A駅"]

    assert {"lat", "lon"}.issubset(result.columns)
    assert row["lat"] == pytest.approx(35.69)  # (35.68 + 35.70) / 2
    assert row["lon"] == pytest.approx(139.87)  # (139.86 + 139.88) / 2


def test_aggregate_by_station_coords_ignore_nan() -> None:
    """座標にNaNが含まれても平均計算で無視されること."""
    df = _sample_df()
    df.loc[0, "最寄駅：緯度"] = pd.NA

    result = aggregate_by_station(df)
    row = result.set_index("最寄駅：名称").loc["A駅"]

    # NaNを除いた残り1件(35.70)が代表座標になる
    assert row["lat"] == pytest.approx(35.70)


def test_station_map_summary_adds_representative_categories() -> None:
    """駅単位集計に代表的な物件種類・価格帯（最頻値）が付与されること."""
    result = station_map_summary(_sample_df()).set_index("最寄駅：名称")

    # A駅は2件とも中古マンション等／1000万~5000万、B駅は宅地(建物)／5000万~1億
    assert {"repr_type", "repr_band", "lat", "lon"}.issubset(result.columns)
    assert result.loc["A駅", "repr_type"] == "中古マンション等"
    assert result.loc["A駅", "repr_band"] == "1000万~5000万"
    assert result.loc["B駅", "repr_type"] == "宅地(建物)"


def test_worst_properties_returns_top_n_by_abs_error() -> None:
    """残差絶対値の降順で上位 N 件を返し、表示用列だけに絞られること."""
    # _sample_df: error_yen=[-10, 10, 100] → |error|=[10,10,100]、最大は B駅(13213) の100
    result = worst_properties(_sample_df(), sort_by="abs_error", n=2)

    assert len(result) == 2
    # 1位は |error|=100 の行（B駅・13213）
    assert result.iloc[0]["市区町村コード"] == 13213
    assert result.iloc[0]["error_yen"] == 100
    # 表示用列のセットだけが含まれる
    expected_cols = {
        "種類",
        "市区町村コード",
        "最寄駅：名称",
        "面積（㎡）",
        "築年数",
        "actual_price_yen",
        "pred_price_yen",
        "error_yen",
        "ape_percent",
    }
    assert set(result.columns) == expected_cols
    # 内部一時列はリーク禁止
    assert "_abs_error" not in result.columns


def test_worst_properties_sort_by_ape() -> None:
    """sort_by='ape' で APE 降順上位を返すこと."""
    # ape=[10, 20, 30] → 1位は ape=30 の行（B駅）
    result = worst_properties(_sample_df(), sort_by="ape", n=1)

    assert len(result) == 1
    assert result.iloc[0]["ape_percent"] == 30.0


def test_worst_properties_n_exceeds_returns_all() -> None:
    """n が件数より大きい場合は全件返ること."""
    result = worst_properties(_sample_df(), n=100)
    assert len(result) == 3


def test_worst_properties_raises_on_invalid_sort_by() -> None:
    """未対応のソート列で ValueError を送出すること."""
    with pytest.raises(ValueError, match="未対応"):
        worst_properties(_sample_df(), sort_by="invalid")


def test_ward_map_summary_adds_representative_categories() -> None:
    """行政区集計に代表的な物件種類・価格帯（最頻値）が付与されること."""
    result = ward_map_summary(_sample_df()).set_index("市区町村コード")

    # 13123は2件とも中古マンション等／1000万~5000万、13213は宅地(建物)／5000万~1億
    assert {"repr_type", "repr_band", "count"}.issubset(result.columns)
    assert result.loc[13123, "repr_type"] == "中古マンション等"
    assert result.loc[13123, "repr_band"] == "1000万~5000万"
    assert result.loc[13213, "repr_type"] == "宅地(建物)"
    # 行政区集計には地図座標(lat/lon)を含めない（GeoJSONのgeometryが担うため）
    assert "lat" not in result.columns


def test_aggregate_predictions_raises_when_group_col_missing() -> None:
    """集計キー列が存在しない場合に KeyError を送出すること."""
    with pytest.raises(KeyError):
        aggregate_predictions(_sample_df(), "存在しない列")


def test_aggregate_predictions_raises_when_coord_col_missing() -> None:
    """指定した座標列が存在しない場合に KeyError を送出すること."""
    df = _sample_df().drop(columns=["最寄駅：緯度"])

    with pytest.raises(KeyError):
        aggregate_predictions(df, "最寄駅：名称", coord_cols=("最寄駅：緯度", "最寄駅：経度"))


def test_load_predictions_reads_valid_csv(tmp_path: Path) -> None:
    """正常なCSVを読み込み、必須列を含むデータフレームを返すこと."""
    csv_path = tmp_path / "pred.csv"
    _sample_df().to_csv(csv_path, index=False, encoding="utf-8")

    df = load_predictions(csv_path)

    assert len(df) == 3
    assert {"pred_price_yen", "actual_price_yen", "ape_percent"}.issubset(df.columns)


def test_load_predictions_raises_when_file_missing(tmp_path: Path) -> None:
    """ファイルが存在しない場合に FileNotFoundError を送出すること."""
    with pytest.raises(FileNotFoundError):
        load_predictions(tmp_path / "does_not_exist.csv")


def test_load_predictions_raises_when_required_column_missing(tmp_path: Path) -> None:
    """必須列が欠落している場合に、不足列名を含む KeyError を送出すること."""
    csv_path = tmp_path / "bad.csv"
    _sample_df().drop(columns=["ape_percent"]).to_csv(csv_path, index=False, encoding="utf-8")

    with pytest.raises(KeyError) as exc_info:
        load_predictions(csv_path)
    assert "ape_percent" in str(exc_info.value)


def test_load_predictions_raises_when_error_yen_missing(tmp_path: Path) -> None:
    """P0-2: error_yen が欠落している場合にも KeyError を送出すること."""
    csv_path = tmp_path / "no_error_yen.csv"
    _sample_df().drop(columns=["error_yen"]).to_csv(csv_path, index=False, encoding="utf-8")

    with pytest.raises(KeyError) as exc_info:
        load_predictions(csv_path)
    assert "error_yen" in str(exc_info.value)


def test_default_predictions_path_falls_back_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BI_PREDICTIONS_PATH 未設定時は既定パスを返すこと."""
    monkeypatch.delenv("BI_PREDICTIONS_PATH", raising=False)

    assert default_predictions_path() == _DEFAULT_PRED_PATH


def test_default_predictions_path_uses_env_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BI_PREDICTIONS_PATH 設定時はそのパスを返すこと."""
    custom = tmp_path / "custom.csv"
    monkeypatch.setenv("BI_PREDICTIONS_PATH", str(custom))

    assert default_predictions_path() == custom


def test_load_predictions_uses_env_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """引数なし呼び出し時に BI_PREDICTIONS_PATH のCSVを読むこと."""
    csv_path = tmp_path / "env.csv"
    _sample_df().to_csv(csv_path, index=False, encoding="utf-8")
    monkeypatch.setenv("BI_PREDICTIONS_PATH", str(csv_path))

    df = load_predictions()

    assert len(df) == 3


def test_filter_predictions_no_condition_returns_all() -> None:
    """条件を渡さなければ全件返ること."""
    df = _sample_df()
    assert len(filter_predictions(df)) == len(df)


def test_filter_predictions_by_ward_and_type() -> None:
    """市区町村コードと物件種類で AND 絞り込みできること."""
    df = _sample_df()

    by_ward = filter_predictions(df, ward_codes=[13123])
    assert set(by_ward["市区町村コード"]) == {13123}
    assert len(by_ward) == 2

    by_type = filter_predictions(df, property_types=["宅地(建物)"])
    assert len(by_type) == 1


def test_filter_predictions_by_ranges_and_yamanote() -> None:
    """面積・築年数レンジ、山手線内側フラグで絞り込めること."""
    df = _sample_df()

    in_range = filter_predictions(df, area_range=(50.0, 100.0), age_range=(0.0, 20.0))
    assert len(in_range) == 1  # 面積60・築15 の1件のみ

    inside = filter_predictions(df, yamanote_inside=True)
    assert set(inside["山手線内側"]) == {1}
    assert len(inside) == 2

    outside = filter_predictions(df, yamanote_inside=False)
    assert len(outside) == 1


def test_filter_predictions_does_not_mutate_input() -> None:
    """元のデータフレームを破壊しないこと（コピーを返す）."""
    df = _sample_df()
    before = len(df)
    _ = filter_predictions(df, ward_codes=[13123])
    assert len(df) == before


def test_available_stations_all_when_no_ward() -> None:
    """行政区未指定なら全駅を昇順で返すこと."""
    df = _sample_df()
    assert available_stations(df) == ["A駅", "B駅"]


def test_available_stations_limited_to_selected_wards() -> None:
    """選択した行政区内の駅のみに絞られること."""
    df = _sample_df()

    # 13123 には A駅のみ（2件）が属する
    assert available_stations(df, ward_codes=[13123]) == ["A駅"]
    # 13213 には B駅のみが属する
    assert available_stations(df, ward_codes=[13213]) == ["B駅"]
    # 両区を選べば両駅
    assert available_stations(df, ward_codes=[13123, 13213]) == ["A駅", "B駅"]


def test_price_band_order_sorts_by_amount() -> None:
    """価格帯が文字列順ではなく金額（平均実測価格）の昇順で並ぶこと."""
    df = pd.DataFrame(
        {
            "actual_price_band": ["高", "低", "中", "低"],
            "actual_price_yen": [9000, 100, 500, 200],
        }
    )
    # 平均実測: 低=150, 中=500, 高=9000 → 金額昇順
    assert price_band_order(df) == ["低", "中", "高"]


def test_summarize_metrics_computes_kpis() -> None:
    """件数・MAE・RMSE・MAPE・Median APE が計算されること."""
    df = _sample_df()
    metrics = summarize_metrics(df)

    assert metrics["count"] == 3
    # |誤差| = |110-100|, |290-300|, |900-1000| = 10, 10, 100
    assert metrics["mae_yen"] == pytest.approx(40.0)
    assert metrics["mape"] == pytest.approx(20.0)  # (10+20+30)/3
    assert metrics["median_ape"] == pytest.approx(20.0)


def test_summarize_metrics_r2_log_when_columns_present() -> None:
    """logスケール列があれば R²(log) が算出されること."""
    df = _sample_df()
    df["actual_log_price"] = [10.0, 11.0, 12.0]
    df["pred_log_price"] = [10.0, 11.0, 12.0]  # 完全一致 → R²=1.0

    metrics = summarize_metrics(df)
    assert metrics["r2_log"] == pytest.approx(1.0)


def test_summarize_metrics_empty_returns_zero_count() -> None:
    """空データでは件数0・指標NaNを返し、例外を出さないこと."""
    empty = _sample_df().iloc[0:0]
    metrics = summarize_metrics(empty)

    assert metrics["count"] == 0
    assert math.isnan(metrics["mae_yen"])
    assert math.isnan(metrics["mape"])
