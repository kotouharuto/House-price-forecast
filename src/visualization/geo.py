"""地理境界・座標GeoJSONの読み込みと検証（UI非依存）.

3種類のGeoJSONを扱う:

- ``load_municipality_geojson``: 行政区ポリゴン（国土数値情報 N03 等の外部データ）。
  P3コロプレス地図で利用。
- ``load_station_geojson``: 駅単位Point（``scripts/csv_to_geojson.py`` の出力）。
- ``load_property_geojson``: 物件単位Point（``scripts/csv_to_geojson.py`` の出力）。

いずれも features の存在＋必須プロパティ（＋必要に応じて geometry 型）を検証する。
"""

import json
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# プロジェクトルート基準の絶対パス（呼び出し場所に依存しないようにする）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 既定の入出力パス（呼び出し側で上書き可能）
_DEFAULT_MUNICIPALITY_GEOJSON_PATH = _PROJECT_ROOT / "configs" / "tokyo_municipalities.geojson"
_DEFAULT_STATION_GEOJSON_PATH = _PROJECT_ROOT / "outputs" / "test_predictions_stations.geojson"
_DEFAULT_PROPERTY_GEOJSON_PATH = _PROJECT_ROOT / "outputs" / "test_predictions_properties.geojson"

# 国土数値情報 N03（行政区域）の市区町村コードプロパティ名（標準）
DEFAULT_CODE_PROPERTY = "N03_007"
# scripts/csv_to_geojson.py の駅GeoJSONで識別子として使うプロパティ
STATION_NAME_PROPERTY = "最寄駅：名称"
# scripts/csv_to_geojson.py の物件GeoJSONで行政区を引くために使うプロパティ
WARD_CODE_PROPERTY = "市区町村コード"


def _read_feature_collection(path: Path) -> dict[str, Any]:
    """ファイルからGeoJSONを読み込み、``features`` が空でないことを確認する."""
    if not path.exists():
        logger.error(f"GeoJSONが見つかりません: {path}")
        raise FileNotFoundError(f"GeoJSONが見つかりません: {path}")

    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    if not data.get("features"):
        logger.error(f"GeoJSONにfeaturesが含まれていません: {path}")
        raise ValueError(f"GeoJSONにfeaturesが含まれていません: {path}")

    return data


def _validate_property(features: list[dict[str, Any]], property_name: str) -> None:
    """全 feature の ``properties`` に ``property_name`` が含まれることを検証する."""
    missing = [
        idx for idx, feat in enumerate(features) if property_name not in feat.get("properties", {})
    ]
    if missing:
        logger.error(
            f"GeoJSONの一部featureに必須プロパティがありません: "
            f"property={property_name}, 件数={len(missing)}"
        )
        raise KeyError(
            f"GeoJSONのfeaturesに必須プロパティ '{property_name}' が無いものが "
            f"{len(missing)} 件あります（最初: index={missing[0]}）"
        )


def _validate_geometry_type(features: list[dict[str, Any]], expected_type: str) -> None:
    """全 feature の ``geometry.type`` が ``expected_type`` と一致することを検証する."""
    mismatched = [
        idx
        for idx, feat in enumerate(features)
        if (feat.get("geometry") or {}).get("type") != expected_type
    ]
    if mismatched:
        logger.error(
            f"GeoJSONに geometry.type が '{expected_type}' 以外の feature があります: "
            f"件数={len(mismatched)}"
        )
        raise ValueError(
            f"GeoJSONに geometry.type が '{expected_type}' でない feature が "
            f"{len(mismatched)} 件あります（最初: index={mismatched[0]}）"
        )


def load_municipality_geojson(
    file_path: str | Path = _DEFAULT_MUNICIPALITY_GEOJSON_PATH,
    *,
    code_property: str = DEFAULT_CODE_PROPERTY,
) -> dict[str, Any]:
    """東京都市区町村のGeoJSON（行政区ポリゴン）を読み込み、市区町村コードプロパティの存在を検証する.

    Args:
        file_path: GeoJSONファイルのパス。デフォルトは
            ``configs/tokyo_municipalities.geojson``。
        code_property: ``features[].properties`` に含まれる市区町村コードの
            プロパティ名。デフォルトは国土数値情報 N03 の ``N03_007``。

    Returns:
        GeoJSON のディクショナリ。

    Raises:
        FileNotFoundError: GeoJSONファイルが存在しない場合。
        ValueError: GeoJSONに ``features`` が含まれていない場合。
        KeyError: いずれかの feature に ``code_property`` が無い場合。
    """
    path = Path(file_path)
    data = _read_feature_collection(path)
    _validate_property(data["features"], code_property)
    logger.info(f"行政区GeoJSONを読み込みました: features={len(data['features'])}, path={path}")
    return data


def load_station_geojson(
    file_path: str | Path = _DEFAULT_STATION_GEOJSON_PATH,
    *,
    key_property: str = STATION_NAME_PROPERTY,
) -> dict[str, Any]:
    """駅単位 Point GeoJSON（``scripts/csv_to_geojson.py`` の出力）を読み込み検証する.

    全 feature が ``geometry.type == "Point"`` かつ ``key_property`` を持つことを確認する。

    Args:
        file_path: GeoJSONファイルのパス。デフォルトは
            ``outputs/test_predictions_stations.geojson``。
        key_property: 駅を識別するプロパティ名。デフォルトは ``最寄駅：名称``。

    Returns:
        GeoJSON のディクショナリ。

    Raises:
        FileNotFoundError: GeoJSONファイルが存在しない場合。
        ValueError: ``features`` が空、または Point 以外の geometry が含まれる場合。
        KeyError: いずれかの feature に ``key_property`` が無い場合。
    """
    path = Path(file_path)
    data = _read_feature_collection(path)
    _validate_geometry_type(data["features"], "Point")
    _validate_property(data["features"], key_property)
    logger.info(f"駅GeoJSONを読み込みました: features={len(data['features'])}, path={path}")
    return data


def load_property_geojson(
    file_path: str | Path = _DEFAULT_PROPERTY_GEOJSON_PATH,
    *,
    key_property: str = WARD_CODE_PROPERTY,
) -> dict[str, Any]:
    """物件単位 Point GeoJSON（``scripts/csv_to_geojson.py`` の出力）を読み込み検証する.

    全 feature が ``geometry.type == "Point"`` かつ ``key_property`` を持つことを確認する。
    ``key_property`` は行政区コード等、物件をグルーピング/参照する際のキーを想定する。

    Args:
        file_path: GeoJSONファイルのパス。デフォルトは
            ``outputs/test_predictions_properties.geojson``。
        key_property: 物件を参照するキープロパティ名。デフォルトは ``市区町村コード``。

    Returns:
        GeoJSON のディクショナリ。

    Raises:
        FileNotFoundError: GeoJSONファイルが存在しない場合。
        ValueError: ``features`` が空、または Point 以外の geometry が含まれる場合。
        KeyError: いずれかの feature に ``key_property`` が無い場合。
    """
    path = Path(file_path)
    data = _read_feature_collection(path)
    _validate_geometry_type(data["features"], "Point")
    _validate_property(data["features"], key_property)
    logger.info(f"物件GeoJSONを読み込みました: features={len(data['features'])}, path={path}")
    return data
