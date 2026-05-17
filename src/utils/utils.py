"""ユーティリティ関数を提供するモジュール."""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(file_path: str, encoding: str = "cp932") -> pd.DataFrame:
    """CSVファイルからデータをロードする.

    Args:
        file_path: CSVファイルのパス。
        encoding: ファイルの文字エンコーディング。デフォルトは ``cp932``
            （国土交通省「不動産情報ライブラリ」のCSVに合わせる）。
            UTF-8の駅データ等を読む場合は ``"utf-8"`` を指定。

    Returns:
        ロードしたデータフレーム。

    Raises:
        FileNotFoundError: 指定したCSVファイルが存在しない場合。
        UnicodeDecodeError: 指定エンコーディングでデコードできないバイトが
            含まれていた場合。
        pd.errors.ParserError: CSVの構造が壊れていてパースに失敗した場合。
    """
    logger.info(f"Loading data from {file_path} (encoding={encoding})...")
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        logger.info(f"Data loaded successfully with shape {df.shape}.")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except UnicodeDecodeError:
        logger.error(f"Failed to decode {file_path} with {encoding} encoding.")
        raise
    except pd.errors.ParserError:
        logger.error(f"Failed to parse CSV file: {file_path}")
        raise
