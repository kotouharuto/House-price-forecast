"""データ前処理および特徴量エンジニアリングを行うモジュール."""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_fontja  # noqa: F401

from utils.logger import get_logger

# ロガーの初期化
logger = get_logger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """CSVファイルからデータをロードする.

    Args:
        file_path: CSVファイルのパス。

    Returns:
        ロードしたデータフレーム。
    """
    logger.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path, encoding='cp932')
        logger.info(f"Data loaded successfully with shape {df.shape}.")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


import pandas as pd


def infer_residential_usage(df: pd.DataFrame, score_threshold: int = 2) -> pd.DataFrame:
    """用途が欠損している行を、他列から推定して『住宅』で補完する。

    判定スコア（各1点、合計が score_threshold 以上で住宅とみなす）:
      - 間取りが記入されている
      - 建物の構造が住宅系（ＲＣ／ＳＲＣ／木造／鉄骨造 など）
      - 種類が住宅系（中古マンション等 など）

    Args:
        df: 不動産情報ライブラリのDataFrame。
        score_threshold: 住宅とみなす最小スコア。デフォルト2。

    Returns:
        用途列を補完した新しいDataFrame。
    """
    # 副作用を避けるためコピー
    result = df.copy()

    # 欠損行のマスク
    mask_missing = result['用途'].isna()

    # 各住宅指標（True=住宅らしい）をベクトル化で算出
    has_madori = result['間取り'].notna()

    residential_structures = ['ＲＣ', 'ＳＲＣ', '木造', '鉄骨造', '軽量鉄骨造', 'ブロック造']
    has_residential_structure = result['建物の構造'].isin(residential_structures)

    residential_types = ['中古マンション等', '宅地(建物)']
    has_residential_type = result['種類'].isin(residential_types)

    # スコア合算
    score = (
        has_madori.astype(int)
        + has_residential_structure.astype(int)
        + has_residential_type.astype(int)
    )
    is_likely_residential = score >= score_threshold

    # 「欠損 かつ 住宅らしい」行だけを補完
    fill_mask = mask_missing & is_likely_residential
    result.loc[fill_mask, '用途'] = '住宅'

    # 補完結果のログ
    n_filled = int(fill_mask.sum())
    n_remaining = int(mask_missing.sum() - n_filled)
    print(f'住宅と推定して補完: {n_filled:,} 件')
    print(f'判定不能のため欠損のまま: {n_remaining:,} 件')

    return result


import pandas as pd


def fill_zoning_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """建蔽率・容積率を立地ベースで階層的に補完する。

    補完の優先順位:
        1. 同一地区名の最頻値（最も近い物件群で補完）
        2. 同一市区町村名の最頻値（地区不明の場合）
        3. 全体の最頻値（最終フォールバック）

    Args:
        df: 不動産情報ライブラリのDataFrame。

    Returns:
        建蔽率・容積率を補完した新しいDataFrame。
    """
    result = df.copy()
    target_cols = ['建蔽率（％）', '容積率（％）']

    for col in target_cols:
        # 1. 地区名グループの最頻値で埋める
        result[col] = result.groupby('地区名')[col].transform(
            lambda s: s.fillna(s.mode().iloc[0]) if not s.mode().empty else s
        )
        # 2. まだ欠損なら市区町村名グループの最頻値で埋める
        result[col] = result.groupby('市区町村名')[col].transform(
            lambda s: s.fillna(s.mode().iloc[0]) if not s.mode().empty else s
        )
        # 3. それでも欠損なら全体の最頻値で埋める
        global_mode = result[col].mode().iloc[0]
        result[col] = result[col].fillna(global_mode)

        n_remaining = int(result[col].isna().sum())
        print(f'{col}: 補完後の欠損数 = {n_remaining:,}')

    return result


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """データのクリーニングを行う.

    - 不要な列の削除
    - 欠損値の処理
    - データ型の変換

    Args:
        df: ロードしたデータフレーム。

    Returns:
        クリーニング後のデータフレーム。
    """
    logger.info("Starting data cleaning...")
    
    # 例: 不要な列 'ID' を削除
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        logger.info("Dropped column 'ID'.")
    
    if "最寄駅：名称" in df.columns:
        missing_station_count = df['最寄駅：名称'].isna().sum()
        logger.info(f"Found {missing_station_count} missing values in '最寄駅：名称' column.")
        # 欠損値を 'Unknown' で埋める
        df['最寄駅：名称'] = df['最寄駅：名称'].fillna('Unknown')
        logger.info("Filled missing values in '最寄駅：名称' with 'Unknown'.")

    if "最寄駅：距離（分）" in df.columns:
        missing_distance_count = df['最寄駅：距離（分）'].isna().sum()
        logger.info(f"Found {missing_distance_count} missing values in '最寄駅：距離（分）' column.")
        # 欠損値を中央値で埋める
        median_distance = df['最寄駅：距離（分）'].median()
        df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].fillna(median_distance)
        logger.info(f"Filled missing values in '最寄駅：距離（分）' with median value {median_distance}.")
    
    if "間取り" in df.columns:
        missing_layout_count = df['間取り'].isna().sum()
        logger.info(f"Found {missing_layout_count} missing values in '間取り' column.")
        # 欠損値を 'Unknown' で埋める
        df['間取り'] = df['間取り'].fillna('Unknown')
        logger.info("Filled missing values in '間取り' with 'Unknown'.")

    if "築年数" in df.columns:
        missing_age_count = df['築年数'].isna().sum()
        logger.info(f"Found {missing_age_count} missing values in '築年数' column.")
        # 欠損値を中央値で埋める
        median_age = df['築年数'].median()
        df['築年数'] = df['築年数'].fillna(median_age)
        logger.info(f"Filled missing values in '築年数' with median value {median_age}.")

    if "建物の構造" in df.columns:
        missing_structure_count = df['建物の構造'].isna().sum()
        logger.info(f"Found {missing_structure_count} missing values in '建物の構造' column.")
        # 欠損値を 'Unknown' で埋める
        df['建物の構造'] = df['建物の構造'].fillna('Unknown')
        logger.info("Filled missing values in '建物の構造' with 'Unknown'.")

    if "用途" in df.columns:
        missing_usage_count = df['用途'].isna().sum()
        logger.info(f"Found {missing_usage_count} missing values in '用途' column.")
        # 用途が欠損している行を推定して補完
        df = infer_residential_usage(df)

    if "今後の利用目的" in df.columns:
        missing_feature_count = df['今後の利用目的'].isna().sum()
        logger.info(f"Found {missing_feature_count} missing values in '今後の利用目的' column.")
        # 欠損値を最頻値で補完(クロス集計結果から)
        most_common_purpose = df['今後の利用目的'].mode()[0]
        df['今後の利用目的'] = df['今後の利用目的'].fillna(most_common_purpose)
        logger.info(f"Filled missing values in '今後の利用目的' with most common value '{most_common_purpose}'.")

    if "都市計画" in df.columns:
        missing_urban_planning_count = df['都市計画'].isna().sum()
        logger.info(f"Found {missing_urban_planning_count} missing values in '都市計画' column.")
        # 欠損値を 'Unknown' で埋める
        df['都市計画'] = df['都市計画'].fillna('Unknown')
        logger.info("Filled missing values in '都市計画' with 'Unknown'.")

    if ["建蔽率（％）", "容積率（％）", "地区名", "市区町村名"] in df.columns:
        # 建蔽率・容積率を立地ベースで補完
        df = fill_zoning_ratios(df)

    if "改装" in df.columns:
        missing_renovation_count = df['改装'].isna().sum()
        logger.info(f"Found {missing_renovation_count} missing values in '改装' column.")
        # 欠損値を 'Unknown' で埋める
        df['改装'] = df['改装'].fillna('Unknown')
        logger.info("Filled missing values in '改装' with 'Unknown'.")

    if "取引の事情等" in df.columns:
        # 列を削除
        df = df.drop(columns=['取引の事情等'])
        logger.info("Dropped column '取引の事情等'.")

    # 例: 日付列 'Date' を datetime 型に変換
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        logger.info("Converted 'Date' column to datetime.")

    logger.info("Data cleaning completed.")
    return df