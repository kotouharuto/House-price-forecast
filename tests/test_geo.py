"""GeoJSONローダ（src.visualization.geo）のテスト."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.visualization.geo import (
    DEFAULT_CODE_PROPERTY,
    STATION_NAME_PROPERTY,
    WARD_CODE_PROPERTY,
    load_municipality_geojson,
    load_property_geojson,
    load_station_geojson,
)


def _sample_geojson(code_property: str = DEFAULT_CODE_PROPERTY) -> dict[str, Any]:
    """テスト用の最小GeoJSON（features=2）を生成する."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {code_property: "13101", "N03_004": "千代田区"},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]},
            },
            {
                "type": "Feature",
                "properties": {code_property: "13102", "N03_004": "中央区"},
                "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [1, 2], [2, 2], [1, 1]]]},
            },
        ],
    }


def _write_geojson(path: Path, data: dict[str, Any]) -> None:
    """テストフィクスチャをファイルへ書き出す."""
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_municipality_geojson_returns_dict(tmp_path: Path) -> None:
    """正常なGeoJSONを読み込み、featuresを含む辞書を返すこと."""
    geo_path = tmp_path / "wards.geojson"
    _write_geojson(geo_path, _sample_geojson())

    data = load_municipality_geojson(geo_path)

    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 2
    assert data["features"][0]["properties"][DEFAULT_CODE_PROPERTY] == "13101"


def test_load_municipality_geojson_raises_when_file_missing(tmp_path: Path) -> None:
    """ファイルが存在しない場合に FileNotFoundError を送出すること."""
    with pytest.raises(FileNotFoundError):
        load_municipality_geojson(tmp_path / "does_not_exist.geojson")


def test_load_municipality_geojson_raises_when_features_empty(tmp_path: Path) -> None:
    """features が空の場合に ValueError を送出すること."""
    geo_path = tmp_path / "empty.geojson"
    _write_geojson(geo_path, {"type": "FeatureCollection", "features": []})

    with pytest.raises(ValueError, match="features"):
        load_municipality_geojson(geo_path)


def test_load_municipality_geojson_raises_when_code_property_missing(tmp_path: Path) -> None:
    """指定したコードプロパティが欠落している場合に KeyError を送出すること."""
    # コードプロパティ名を別物に変えたGeoJSONを用意し、デフォルト名で読み込ませる
    geo_path = tmp_path / "wrong_property.geojson"
    _write_geojson(geo_path, _sample_geojson(code_property="WRONG_KEY"))

    with pytest.raises(KeyError, match=DEFAULT_CODE_PROPERTY):
        load_municipality_geojson(geo_path)


def test_load_municipality_geojson_accepts_custom_code_property(tmp_path: Path) -> None:
    """code_property 引数で別プロパティ名のGeoJSONも読み込めること."""
    geo_path = tmp_path / "custom.geojson"
    _write_geojson(geo_path, _sample_geojson(code_property="city_code"))

    data = load_municipality_geojson(geo_path, code_property="city_code")

    assert len(data["features"]) == 2


# ---------- 駅 Point GeoJSON ローダ ----------


def _sample_station_geojson(
    key_property: str = STATION_NAME_PROPERTY,
) -> dict[str, Any]:
    """テスト用の最小駅GeoJSON（Point×2）."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {key_property: "A駅", "count": 2},
                "geometry": {"type": "Point", "coordinates": [139.86, 35.69]},
            },
            {
                "type": "Feature",
                "properties": {key_property: "B駅", "count": 1},
                "geometry": {"type": "Point", "coordinates": [139.47, 35.74]},
            },
        ],
    }


def test_load_station_geojson_returns_dict(tmp_path: Path) -> None:
    """正常な駅GeoJSONを読み込み、Point featureを返すこと."""
    geo_path = tmp_path / "stations.geojson"
    _write_geojson(geo_path, _sample_station_geojson())

    data = load_station_geojson(geo_path)

    assert len(data["features"]) == 2
    assert data["features"][0]["geometry"]["type"] == "Point"


def test_load_station_geojson_raises_when_geometry_not_point(tmp_path: Path) -> None:
    """Point以外の geometry が含まれる場合に ValueError を送出すること."""
    bad = _sample_station_geojson()
    bad["features"][1]["geometry"] = {"type": "Polygon", "coordinates": [[[0, 0]]]}
    geo_path = tmp_path / "stations_bad.geojson"
    _write_geojson(geo_path, bad)

    with pytest.raises(ValueError, match="Point"):
        load_station_geojson(geo_path)


def test_load_station_geojson_raises_when_key_property_missing(tmp_path: Path) -> None:
    """key_property が無い場合に KeyError を送出すること."""
    geo_path = tmp_path / "stations_wrong.geojson"
    _write_geojson(geo_path, _sample_station_geojson(key_property="WRONG"))

    with pytest.raises(KeyError, match=STATION_NAME_PROPERTY):
        load_station_geojson(geo_path)


# ---------- 物件 Point GeoJSON ローダ ----------


def _sample_property_geojson(
    key_property: str = WARD_CODE_PROPERTY,
) -> dict[str, Any]:
    """テスト用の最小物件GeoJSON（Point×2）."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {key_property: 13123, "種類": "中古マンション等"},
                "geometry": {"type": "Point", "coordinates": [139.86, 35.69]},
            },
            {
                "type": "Feature",
                "properties": {key_property: 13213, "種類": "宅地(建物)"},
                "geometry": {"type": "Point", "coordinates": [139.47, 35.74]},
            },
        ],
    }


def test_load_property_geojson_returns_dict(tmp_path: Path) -> None:
    """正常な物件GeoJSONを読み込み、Point featureを返すこと."""
    geo_path = tmp_path / "properties.geojson"
    _write_geojson(geo_path, _sample_property_geojson())

    data = load_property_geojson(geo_path)

    assert len(data["features"]) == 2
    assert data["features"][0]["properties"][WARD_CODE_PROPERTY] == 13123


def test_load_property_geojson_raises_when_key_property_missing(tmp_path: Path) -> None:
    """key_property が無い場合に KeyError を送出すること."""
    geo_path = tmp_path / "properties_wrong.geojson"
    _write_geojson(geo_path, _sample_property_geojson(key_property="WRONG"))

    with pytest.raises(KeyError, match=WARD_CODE_PROPERTY):
        load_property_geojson(geo_path)
