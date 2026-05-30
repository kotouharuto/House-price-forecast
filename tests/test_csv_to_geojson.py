"""scripts/csv_to_geojson.py のテスト."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/ を import 可能にするためプロジェクトルートを sys.path に追加
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.csv_to_geojson import (  # noqa: E402
    _to_native,
    build_property_features,
    build_station_features,
)
from src.visualization.aggregate import STATION_LAT_COL, STATION_LON_COL  # noqa: E402


def _full_sample_df() -> pd.DataFrame:
    """build_station_features まで通せる、必須列を満たした最小サンプル."""
    return pd.DataFrame(
        {
            "市区町村コード": [13123, 13123, 13213],
            "最寄駅：名称": ["A駅", "A駅", "B駅"],
            STATION_LAT_COL: [35.68, 35.70, 35.74],
            STATION_LON_COL: [139.86, 139.88, 139.47],
            "種類": ["中古マンション等", "中古マンション等", "宅地(建物)"],
            "actual_price_band": ["1000万~5000万", "1000万~5000万", "5000万~1億"],
            "面積（㎡）": [40.0, 60.0, 80.0],
            "築年数": [5.0, 15.0, 30.0],
            "山手線内側": [1, 1, 0],
            "pred_price_yen": [100, 300, 1000],
            "actual_price_yen": [110, 290, 900],
            "error_yen": [-10, 10, 100],
            "ape_percent": [10.0, 20.0, 30.0],
        }
    )


def test_to_native_handles_nan_and_numpy_types() -> None:
    """NaN/None/numpy型を JSON 可能なネイティブ値に変換すること."""
    assert _to_native(None) is None
    assert _to_native(float("nan")) is None
    assert _to_native(pd.NA) is None
    assert _to_native(np.int64(7)) == 7
    assert _to_native(np.float64(1.5)) == 1.5
    assert _to_native(True) is True  # bool は int に潰さない
    assert _to_native("hello") == "hello"
    val = _to_native(np.float64(3.14))
    assert isinstance(val, float) and not math.isnan(val)


def test_build_property_features_excludes_missing_coords_and_drops_coord_props() -> None:
    """座標欠損行は除外され、座標列は properties に含まれないこと."""
    df = pd.DataFrame(
        {
            STATION_LAT_COL: [35.68, None, 35.70],
            STATION_LON_COL: [139.86, 139.88, None],
            "name": ["a", "b", "c"],
        }
    )

    features = build_property_features(df)

    # 座標どちらか欠損の2行は除外、有効1件のみ
    assert len(features) == 1
    feat = features[0]
    # GeoJSON仕様は [lon, lat] の順
    assert feat["geometry"] == {"type": "Point", "coordinates": [139.86, 35.68]}
    assert feat["properties"] == {"name": "a"}
    assert STATION_LAT_COL not in feat["properties"]
    assert STATION_LON_COL not in feat["properties"]


def test_build_station_features_aggregates_and_drops_coord_props() -> None:
    """駅単位の集計が Point Feature 化され、lat/lon は properties から除かれること."""
    features = build_station_features(_full_sample_df())

    # A駅(2件)とB駅(1件)の2 Feature
    assert len(features) == 2
    by_station = {feat["properties"]["最寄駅：名称"]: feat for feat in features}

    a = by_station["A駅"]
    assert a["geometry"]["type"] == "Point"
    # A駅の代表座標は緯度経度のグループ平均
    assert a["geometry"]["coordinates"] == [pytest_approx(139.87), pytest_approx(35.69)]
    # 集計列は properties に含まれる、座標列は含まれない
    assert a["properties"]["count"] == 2
    assert "lat" not in a["properties"]
    assert "lon" not in a["properties"]
    assert a["properties"]["repr_type"] == "中古マンション等"


# pytest.approx を逐一importしないための小ヘルパー（軽量比較用）
def pytest_approx(value: float, tol: float = 1e-6) -> "_Approx":
    return _Approx(value, tol)


class _Approx:
    """浮動小数の近似比較ヘルパー（テスト内専用）."""

    def __init__(self, value: float, tol: float) -> None:
        self.value = value
        self.tol = tol

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and abs(other - self.value) <= self.tol

    def __repr__(self) -> str:
        return f"~{self.value}"
