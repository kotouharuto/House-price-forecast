"""特徴量エンジニアリングを行うモジュール."""

# 標準ライブラリ
import sys
import time
from pathlib import Path

# サードパーティ
import numpy as np
import pandas as pd
import requests

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# プロジェクト内モジュール（sys.path 操作後である必要があるため E402 を許容）
from src.utils.logger import get_logger  # noqa: E402
from src.utils.utils import load_data  # noqa: E402

# モジュール定数
# 生データの配置先（プロジェクトルート基準の絶対パスで固定）
_DATA_RAW_DIR = _PROJECT_ROOT / "data" / "raw"

# ロガーの初期化
logger = get_logger(__name__)


def get_lat_lon(address: str) -> tuple[float | None, float | None]:
    """住所文字列から緯度・経度を取得する（国土地理院APIを使用）.

    Args:
        address: ジオコーディング対象の住所文字列。

    Returns:
        ``(緯度, 経度)`` のタプル。取得失敗時は ``(None, None)``。
    """
    url = f"https://msearch.gsi.go.jp/address-search/AddressSearch?q={address}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json()

        if results:
            # APIの仕様上 [経度(lon), 緯度(lat)] の順で返ってくるため注意
            lon, lat = results[0]["geometry"]["coordinates"]
            # 連続アクセスを避けるため1リクエストごとに待機（規約配慮）
            time.sleep(0.2)
            return lat, lon
        return None, None

    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        logger.warning(f"住所のジオコーディングに失敗 ({address}): {exc}")
        return None, None


def merge_station_data(df: pd.DataFrame) -> pd.DataFrame:
    """最寄駅の情報を結合する.

    Args:
        df: データフレーム。

    Returns:
        最寄駅の情報が結合されたデータフレーム。
    """
    # 駅データは UTF-8 で配布されているため明示的に指定する
    # （プロジェクトルート基準の絶対パスで指定し、呼び出し場所に依存しないようにする）
    station_df = load_data(str(_DATA_RAW_DIR / "station_data_2026.csv"), encoding="utf-8")

    # 不動産情報ライブラリ側のカラム名に揃える（DataFrame.rename で実際に列名を変更する）
    station_df = station_df.rename(
        columns={
            "station_name": "最寄駅：名称",
            "lat": "最寄駅：緯度",
            "lon": "最寄駅：経度",
        }
    )

    # 同名駅が複数路線にまたがるため重複を除去（先頭の1行を残す）
    station_df = station_df.drop_duplicates(subset=["最寄駅：名称"], keep="first")

    # 必要な3列のみに絞ってから merge（不要な列が混入するのを防ぐ）
    df = df.merge(
        station_df[["最寄駅：名称", "最寄駅：緯度", "最寄駅：経度"]],
        on="最寄駅：名称",
        how="left",
    )
    logger.info("Merged station data with main dataframe.")

    return df


def categorize_zoning(zoning):
    """「都市計画」をカテゴリに分類するモジュール."""
    if zoning == 'Unknown':
        return 'Unknown'
    elif '商業' in zoning:
        return '商業系' # 商業、近隣商業など
    elif '工業' in zoning:
        return '工業系' # 工業、準工業など
    elif '住' in zoning:
        return '住居系' # 1中住専、1種住居など
    else:
        return 'その他'


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量の変換を行う.最低限分析が行える状態にするための処理を実装するモジュール.

    Args:
        df: 前処理されたデータフレーム。

    Returns:
        特徴量エンジニアリングされたデータフレーム。
    """

    # 価格情報区分を数値化
    if "価格情報区分" in df.columns:
        df["価格情報区分"] = df["価格情報区分"].map({"成約価格情報": 1, "不動産取引価格情報": 0})
        logger.info("Encoded '価格情報区分' column to binary values.")

    # フルの住所を作成、緯度経度を取得（必要な列がすべて揃っている時のみ実行）
    address_cols = ["都道府県名", "市区町村名"]
    if all(col in df.columns for col in address_cols):
        df["住所"] = df["都道府県名"] + df["市区町村名"]

        # 同一住所がDataFrame中に多数あるため、ユニークな住所だけAPIを叩いてキャッシュする
        unique_addresses = df["住所"].dropna().unique()
        logger.info(f"Geocoding {len(unique_addresses)} unique addresses via 国土地理院API...")
        address_to_latlon = {addr: get_lat_lon(addr) for addr in unique_addresses}

        # 取得結果をDataFrameに展開
        df["緯度"] = df["住所"].map(lambda a: address_to_latlon.get(a, (None, None))[0])
        df["経度"] = df["住所"].map(lambda a: address_to_latlon.get(a, (None, None))[1])

        # 都道府県名・市区町村名は削除（住所列に集約済み）
        df = df.drop(columns=address_cols)

        logger.info("Created '住所' column and obtained '緯度' and '経度'.")

    if "間取り" in df.columns:
        df["room_count"] = df["間取り"].str.extract(r"(\d+)")[0].astype(float)
        df["has_L"] = df["間取り"].str.contains("L").astype(int)
        df["has_D"] = df["間取り"].str.contains("D").astype(int)
        df["has_K"] = df["間取り"].str.contains("K").astype(int)
        df["has_S"] = df["間取り"].str.contains("S").astype(int)
        df = df.drop(columns=["間取り"])

        logger.info("Engineered features from '間取り' column.")

    """文字列特徴量のカテゴリ化"""
    if "建物の構造" in df.columns:
        df["建物の構造"] = df["建物の構造"].astype("category")
        logger.info("Converted '建物の構造' column to categorical type.")

    if "用途" in df.columns:
        df["用途"] = df["用途"].astype("category")
        logger.info("Converted '用途' column to categorical type.")

    if "今後の利用目的" in df.columns:
        df["今後の利用目的"] = df["今後の利用目的"].astype("category")
        logger.info("Converted '今後の利用目的' column to categorical type.")

    if "都市計画" in df.columns:
        df["都市計画"] = df["都市計画"].replace({"工業専用": "工業"})
        df["都市計画"] = df["都市計画"].apply(categorize_zoning)
        logger.info("Categorized '都市計画' column.")


    if "取引時期" in df.columns:
        period_pattern = r"(?P<取引年>\d{4})年?第(?P<取引四半期>[1-4])四半期"

        df[["取引年", "取引四半期"]] = (
            df["取引時期"]
            .astype(str)
            .str.extract(period_pattern)
        )

        df["取引四半期"] = pd.to_numeric(df["取引四半期"], errors="coerce").astype("Int64")
        logger.info("Extracted '取引年' and '取引四半期' from '取引時期' column.")

        df.drop(columns=["取引時期", "取引年"], inplace=True)
        logger.info("Dropped '取引時期' and '取引年' columns.")

    if "改装" in df.columns:
        mapping = {
            "改装済み": 1,
            "未改装": 0,
            "Unknown": np.nan,
        }

        df["改装"] = df["改装"].map(mapping)
        logger.info("Encoded '改装' column to binary values.")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量エンジニアリングを実行するモジュール.
    Args:
        df: データフレーム.
    Returns:
        特徴量エンジニアリングを施したデータフレーム.
    """

    df = merge_station_data(df)
    df = build_features(df)

    return df
