"""予測結果CSVを物件単位・駅単位の2種類のGeoJSONに変換するスクリプト.

物件単位 (``*_properties.geojson``): 各行をPoint Feature化（最寄駅の緯度経度）。
駅単位 (``*_stations.geojson``):     ``station_map_summary`` の集計結果をPoint Feature化。

入力CSVは事前にコピーしてから変換する（コピー先は出力ディレクトリに ``*_copy.csv``）。

使い方:
    uv run python scripts/csv_to_geojson.py
    uv run python scripts/csv_to_geojson.py \\
        --src outputs/test_predictions.csv --out-dir outputs
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import get_logger  # noqa: E402
from src.visualization.aggregate import (  # noqa: E402
    STATION_LAT_COL,
    STATION_LON_COL,
    load_predictions,
    station_map_summary,
)

logger = get_logger(__name__)

_DEFAULT_SRC = _PROJECT_ROOT / "outputs" / "test_predictions.csv"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "outputs"
_COPY_SUFFIX = "_copy"


def _to_native(value: Any) -> Any:
    """JSONシリアライズ可能なPythonネイティブ型に変換する（NaN/NaTは ``None``）."""
    if value is None:
        return None
    # bool は int サブクラスなので先に分岐し、True/False をそのまま返す
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    # 配列を pd.isna に渡すと曖昧な真理値になるため、スカラのみ判定する
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy scalar → Python scalar
        try:
            return value.item()
        except (ValueError, TypeError):
            return str(value)
    return value


def _row_to_properties(row: pd.Series) -> dict[str, Any]:
    """DataFrame行をGeoJSON properties用のディクショナリに変換する."""
    return {str(col): _to_native(val) for col, val in row.items()}


def _build_point_feature(lat: float, lon: float, properties: dict[str, Any]) -> dict[str, Any]:
    """Point geometry の GeoJSON Feature を生成する（GeoJSONの順序は [lon, lat]）."""
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": properties,
    }


def build_property_features(df: pd.DataFrame) -> list[dict[str, Any]]:
    """予測結果CSVの各行をPoint Featureに変換する.

    最寄駅の緯度/経度が欠損している行は除外する。座標列は properties から
    除外し、geometry との重複を避ける。
    """
    plotted = df.dropna(subset=[STATION_LAT_COL, STATION_LON_COL])
    features: list[dict[str, Any]] = []
    for _, row in plotted.iterrows():
        lat = float(row[STATION_LAT_COL])
        lon = float(row[STATION_LON_COL])
        props = _row_to_properties(row.drop(labels=[STATION_LAT_COL, STATION_LON_COL]))
        features.append(_build_point_feature(lat, lon, props))
    return features


def build_station_features(df: pd.DataFrame) -> list[dict[str, Any]]:
    """駅単位の集計（station_map_summary）をPoint Featureに変換する.

    緯度/経度が欠損する駅は除外する。代表座標 ``lat`` / ``lon`` は geometry に移し、
    properties からは除く。
    """
    summary = station_map_summary(df).dropna(subset=["lat", "lon"])
    features: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        props = _row_to_properties(row.drop(labels=["lat", "lon"]))
        features.append(_build_point_feature(lat, lon, props))
    return features


def _feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Feature の配列を GeoJSON FeatureCollection に包む."""
    return {"type": "FeatureCollection", "features": features}


def _write_geojson(path: Path, fc: dict[str, Any]) -> None:
    """GeoJSONをUTF-8で出力する（非ASCIIを保持）."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    logger.info(f"GeoJSONを出力しました: {path} (features={len(fc['features']):,})")


def convert(src: Path, out_dir: Path) -> dict[str, Path]:
    """CSVをコピーし、物件Point/駅Point の2種類のGeoJSONを出力する.

    Args:
        src: 入力予測結果CSV。
        out_dir: GeoJSONとコピーCSVの出力先ディレクトリ。

    Returns:
        生成したファイルのパス辞書（``copy`` / ``properties`` / ``stations``）。

    Raises:
        FileNotFoundError: 入力CSVが存在しない場合。
    """
    if not src.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 入力CSVをコピー（コピー先を後段の読み込みで使う）
    copy_path = out_dir / f"{src.stem}{_COPY_SUFFIX}{src.suffix}"
    shutil.copy2(src, copy_path)
    logger.info(f"CSVをコピーしました: {src} -> {copy_path}")

    # 2) 読み込み（コピー側）。必須列の検証は load_predictions に委譲
    df = load_predictions(copy_path)

    # 3) 物件 GeoJSON
    properties_path = out_dir / f"{src.stem}_properties.geojson"
    _write_geojson(properties_path, _feature_collection(build_property_features(df)))

    # 4) 駅 GeoJSON
    stations_path = out_dir / f"{src.stem}_stations.geojson"
    _write_geojson(stations_path, _feature_collection(build_station_features(df)))

    return {"copy": copy_path, "properties": properties_path, "stations": stations_path}


def main() -> None:
    """コマンドラインエントリポイント."""
    parser = argparse.ArgumentParser(
        description="予測結果CSVを物件単位・駅単位の2種類のGeoJSONに変換する。"
    )
    parser.add_argument("--src", type=Path, default=_DEFAULT_SRC, help="入力予測結果CSVのパス")
    parser.add_argument(
        "--out-dir", type=Path, default=_DEFAULT_OUT_DIR, help="GeoJSONの出力先ディレクトリ"
    )
    args = parser.parse_args()

    result = convert(args.src, args.out_dir)
    print(f"コピー先:    {result['copy']}")
    print(f"物件GeoJSON: {result['properties']}")
    print(f"駅GeoJSON:   {result['stations']}")


if __name__ == "__main__":
    main()
