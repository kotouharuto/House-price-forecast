"""特徴量エンジニアリングを行うモジュール."""

# 標準ライブラリ
import sys
import time
import unicodedata
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

# 地球の半径（km）。haversine 距離計算に使用
_EARTH_RADIUS_KM = 6371

# 山手線のバウンディングボックス（簡易な内側判定用）
# 北端: 田端(35.738) / 南端: 品川(35.628) / 西端: 新宿(139.700) / 東端: 秋葉原(139.780)
_YAMANOTE_LAT_RANGE = (35.628, 35.738)
_YAMANOTE_LON_RANGE = (139.700, 139.780)

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

    駅名の表記揺れに対応するため:
      - 駅マスタは東京都 (``pref_cd == 13``) の駅のみに絞り込む
      - 不動産DB側の ``"大森(東京)"`` のような接尾辞を除去してから merge する

    Args:
        df: データフレーム。

    Returns:
        最寄駅の緯度経度が結合されたデータフレーム。
    """
    # 駅データは UTF-8 で配布されているため明示的に指定する
    # （プロジェクトルート基準の絶対パスで指定し、呼び出し場所に依存しないようにする）
    station_df = load_data(str(_DATA_RAW_DIR / "station_data_2026.csv"), encoding="utf-8")

    # 駅マスタを東京都 (pref_cd == 13) に絞る
    # （これをしないと「大森」で神奈川県の駅にマッチする等の誤マージが起きる）
    station_df = station_df[station_df["pref_cd"] == 13]

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

    # 不動産DB側の駅名から "(東京)" のような接尾辞を除去（表記揺れ対策）
    df = df.copy()
    df["最寄駅：名称"] = (
        df["最寄駅：名称"].astype(str).str.replace(r"\([^)]*\)", "", regex=True).str.strip()
    )

    # 必要な3列のみに絞ってから merge（不要な列が混入するのを防ぐ）
    df = df.merge(
        station_df[["最寄駅：名称", "最寄駅：緯度", "最寄駅：経度"]],
        on="最寄駅：名称",
        how="left",
    )
    n_missing = int(df["最寄駅：緯度"].isna().sum())
    logger.info(f"Merged station data. Missing coordinates after merge: {n_missing:,} rows.")

    return df


def categorize_zoning(zoning):
    """「都市計画」をカテゴリに分類するモジュール."""
    if zoning == "Unknown":
        return "Unknown"
    elif "商業" in zoning:
        return "商業系"  # 商業、近隣商業など
    elif "工業" in zoning:
        return "工業系"  # 工業、準工業など
    elif "住" in zoning:
        return "住居系"  # 1中住専、1種住居など
    else:
        return "その他"


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2地点間の haversine（大円）距離を km 単位で計算する.

    Args:
        lat1: 地点1の緯度。
        lon1: 地点1の経度。
        lat2: 地点2の緯度。
        lon2: 地点2の経度。

    Returns:
        2地点間の距離（km）。
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return _EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_yamanote_inside_flag(df: pd.DataFrame) -> pd.DataFrame:
    """山手線の内側にある物件にフラグを立てる.

    山手線駅の緯度経度のバウンディングボックスで判定する簡易版。
    北端: 田端(35.738) / 南端: 品川(35.628)
    西端: 新宿(139.700) / 東端: 秋葉原(139.780)
    """
    df = df.copy()
    if {"最寄駅：緯度", "最寄駅：経度"}.issubset(df.columns):
        df["山手線内側"] = (
            df["最寄駅：緯度"].between(*_YAMANOTE_LAT_RANGE)
            & df["最寄駅：経度"].between(*_YAMANOTE_LON_RANGE)
        ).astype(int)
    return df


def categorical_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ変数（文字列列）の表記を正規化する.

    object / string / category 型の列を対象に、以下の正規化を一括適用する:
      1. NFKC正規化: 全角英数記号を半角に統一（"１ＬＤＫ" -> "1LDK"）
      2. 前後の空白を除去（``str.strip``）
      3. 空文字 ``""`` を ``NaN`` に統一
      4. プロジェクト固有の表記揺れを統一（"改装済み" -> "改装済" 等）

    Args:
        df: データフレーム。

    Returns:
        文字列列が正規化された新しいデータフレーム。
    """
    df = df.copy()

    # 対象列を取得（category 型は文字列操作のため一旦 object に戻す）
    string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    category_cols = df.select_dtypes(include=["category"]).columns.tolist()
    for col in category_cols:
        df[col] = df[col].astype(object)

    target_cols = string_cols + category_cols
    for col in target_cols:
        df[col] = (
            df[col]
            .map(lambda s: unicodedata.normalize("NFKC", s) if isinstance(s, str) else s)
            .map(lambda s: s.strip() if isinstance(s, str) else s)
            .replace("", np.nan)
        )

    # プロジェクト固有の表記揺れ吸収
    known_replacements: dict[str, dict[str, str]] = {
        "都市計画": {"工業専用": "工業"},
        "改装": {"改装済み": "改装済"},
    }
    for col, mapping in known_replacements.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    # 元 category 型の列を再度 category に戻す
    for col in category_cols:
        df[col] = df[col].astype("category")

    logger.info(
        f"Categorical normalization applied to {len(target_cols)} columns "
        f"(string: {len(string_cols)}, category: {len(category_cols)})."
    )
    return df


def categorize_features(df: pd.DataFrame) -> pd.DataFrame:
    """カテゴリ変数として扱うべき列を ``category`` 型に一括変換する.

    LightGBM / XGBoost は ``category`` dtype の列を自動的にカテゴリ特徴量として
    扱うため、変換しておくことでモデル学習時の categorical_feature 指定が不要になる。

    Args:
        df: データフレーム。

    Returns:
        対象列が ``category`` 型に変換された新しいデータフレーム。
    """
    df = df.copy()

    # カテゴリ変数として扱う列のリスト（追加・削除はここで一元管理）
    categorical_columns = [
        "種類",
        "最寄駅：名称",
        "建物の構造",
        "用途",
        "今後の利用目的",
        "都市計画",
        "住所",
    ]

    converted = [col for col in categorical_columns if col in df.columns]
    for col in converted:
        df[col] = df[col].astype("category")

    logger.info(f"Converted {len(converted)} columns to category type: {converted}")
    return df


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

    # 位置情報は merge_station_data() で付与した「最寄駅：緯度／経度」と
    # 「最寄駅：距離（分）」で十分代替できるため、住所からのジオコーディングは行わない。
    # 都道府県名・市区町村名・地区名は連結して「住所」カテゴリ列にまとめる。
    address_cols = ["都道府県名", "市区町村名", "地区名"]
    if all(col in df.columns for col in address_cols):
        df["住所"] = (
            df["都道府県名"].astype(str) + df["市区町村名"].astype(str) + df["地区名"].astype(str)
        )
        df["住所"] = df["住所"].astype("category")
        df = df.drop(columns=address_cols)
        logger.info("Combined '都道府県名', '市区町村名', '地区名' into '住所' column.")

    if "間取り" in df.columns:
        # 国土交通省データは間取りが全角（例: "１ＬＤＫ"）で記録されているため
        # NFKC正規化で半角に揃えてから抽出する（"１ＬＤＫ" -> "1LDK"）
        df["間取り"] = df["間取り"].astype(str).map(lambda s: unicodedata.normalize("NFKC", s))

        df["room_count"] = df["間取り"].str.extract(r"(\d+)")[0].astype(float)
        df["has_L"] = df["間取り"].str.contains("L", na=False).astype(int)
        df["has_D"] = df["間取り"].str.contains("D", na=False).astype(int)
        df["has_K"] = df["間取り"].str.contains("K", na=False).astype(int)
        df["has_S"] = df["間取り"].str.contains("S", na=False).astype(int)
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
        df["都市計画"] = df["都市計画"].map(categorize_zoning).astype("category")
        logger.info("Categorized '都市計画' column into broader categories.")

    if "取引時期" in df.columns:
        period_pattern = r"(?P<取引年>\d{4})年?第(?P<取引四半期>[1-4])四半期"

        df[["取引年", "取引四半期"]] = df["取引時期"].astype(str).str.extract(period_pattern)

        df["取引四半期"] = pd.to_numeric(df["取引四半期"], errors="coerce").astype("Int64")
        logger.info("Extracted '取引年' and '取引四半期' from '取引時期' column.")

        df.drop(columns=["取引時期", "取引年"], inplace=True)
        logger.info("Dropped '取引時期' and '取引年' columns.")

    if "改装" in df.columns:
        mapping = {
            "改装済": 1,
            "未改装": 0,
            "Unknown": np.nan,
        }

        df["改装"] = df["改装"].map(mapping)
        logger.info("Encoded '改装' column to binary values.")

    if {"最寄駅：緯度", "最寄駅：経度"}.issubset(df.columns):
        df["最寄駅：緯度"] = pd.to_numeric(df["最寄駅：緯度"], errors="coerce")
        df["最寄駅：経度"] = pd.to_numeric(df["最寄駅：経度"], errors="coerce")
        logger.info("Converted '最寄駅：緯度' and '最寄駅：経度' columns to numeric type.")

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """新規特徴量を生成するモジュール.

    Args:
        df: データフレーム.

    Returns:
        特徴量エンジニアリングを施したデータフレーム.
    """

    # 対数変換した 面積（㎡）
    if "面積（㎡）" in df.columns:
        df["log_面積"] = np.log(df["面積（㎡）"].replace(0, np.nan))
        logger.info("Generated 'log_面積' feature by applying log transformation to '面積（㎡）'.")

    # 対数変換した 最寄駅：距離（分）
    if "最寄駅：距離（分）" in df.columns:
        df["log_最寄駅：距離"] = np.log(df["最寄駅：距離（分）"].replace(0, np.nan))
        logger.info(
            "Generated 'log_最寄駅：距離' feature by applying log transformation to '最寄駅：距離（分）'."
        )

    # 築年数^2 経年劣化の加速度を表現するため
    if "築年数" in df.columns:
        df["築年数^2"] = df["築年数"] ** 2
        logger.info("Generated '築年数^2' feature by squaring the '築年数' column.")

    # 中心地からの距離
    if {"最寄駅：緯度", "最寄駅：経度"}.issubset(df.columns):
        # 東京駅の緯度経度（中心地の代表点として）
        tokyo_lat, tokyo_lon = 35.681236, 139.767125

        df["中心地_距離"] = df.apply(
            lambda row: (
                haversine(row["最寄駅：緯度"], row["最寄駅：経度"], tokyo_lat, tokyo_lon)
                if pd.notnull(row["最寄駅：緯度"]) and pd.notnull(row["最寄駅：経度"])
                else np.nan
            ),
            axis=1,
        )
        logger.info(
            "Generated '中心地_距離' feature by calculating Haversine distance to Tokyo Station."
        )

    # 山手線の内側かどうかを判定
    if {"最寄駅：緯度", "最寄駅：経度"}.issubset(df.columns):
        df["最寄駅：緯度"] = pd.to_numeric(df["最寄駅：緯度"], errors="coerce")
        df["最寄駅：経度"] = pd.to_numeric(df["最寄駅：経度"], errors="coerce")

        # 山手線内側フラグ（都心プレミアム捕捉用）
        df["山手線内側"] = (
            df["最寄駅：緯度"].between(35.628, 35.738)
            & df["最寄駅：経度"].between(139.700, 139.780)
        ).astype(int)

        logger.info("Added '山手線内側' flag.")

    # 建ぺい率、容積率関連
    if {"容積率（％）", "建ぺい率（％）"}.issubset(df.columns):
        df["容積消化率（％）"] = df["容積率（％）"] / df["建ぺい率（％）"]
        df["log_建ぺい率"] = np.log(df["建ぺい率（％）"].replace(0, np.nan))
        df["log_容積率"] = np.log(df["容積率（％）"].replace(0, np.nan))
        logger.info("Added 容積消化率 in dataframe.")

    # 容積率と駅徒歩の組み合わせ
    if {"容積率（％）", "最寄駅：距離（分）"}.issubset(df.columns):
        df["容積距離"] = df["容積率（％）"] * df["最寄駅：距離（分）"]
        logger.info("Added 容積距離 in dataframe")

    # 低額帯(~2000万円未満)の特徴量強化


    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量エンジニアリングを実行するモジュール.
    Args:
        df: データフレーム.
    Returns:
        特徴量エンジニアリングを施したデータフレーム.
    """
    start_time = time.time()

    df = merge_station_data(df)  # 駅情報データと結合
    df = categorical_normalization(df)  # カテゴリ変数の表記を正規化
    df = build_features(df)  # 特徴量の変換を行う
    df = categorize_features(df)  # 該当列を category 型に一括変換
    df = generate_features(df)  # 新規特徴量の生成

    end_time = time.time()
    logger.info(f"Data feature engineering completed in {end_time - start_time:.2f} seconds.")

    return df
