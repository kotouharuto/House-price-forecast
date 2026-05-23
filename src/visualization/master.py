"""地理マスタ（市区町村コード↔名称）を扱うモジュール（UI非依存）.

予測結果には ``市区町村コード``（例: ``13101``）しか含まれないため、BIツールで
人間が読みやすい ``市区町村名``（例: ``千代田区``）を表示・選択できるよう、
コードと名称の対応表を ``configs/tokyo_municipality_master.csv`` から読み込む。
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# プロジェクトルート基準の絶対パス（呼び出し場所に依存しないようにする）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MASTER_PATH = _PROJECT_ROOT / "configs" / "tokyo_municipality_master.csv"

# マスタCSVの列名
CODE_COL = "市区町村コード"
NAME_COL = "市区町村名"


def load_municipality_names(
    master_path: Path = _DEFAULT_MASTER_PATH,
) -> dict[int, str]:
    """市区町村コード → 名称の対応辞書を読み込む.

    Args:
        master_path: マスタCSVのパス。デフォルトは
            ``configs/tokyo_municipality_master.csv``。

    Returns:
        ``{市区町村コード(int): 市区町村名(str)}`` の辞書。

    Raises:
        FileNotFoundError: マスタCSVが存在しない場合。
        KeyError: 必須列（市区町村コード / 市区町村名）が欠落している場合。
    """
    path = Path(master_path)
    if not path.exists():
        logger.error(f"市区町村マスタが見つかりません: {path}")
        raise FileNotFoundError(f"市区町村マスタが見つかりません: {path}")

    df = pd.read_csv(path, encoding="utf-8")

    missing = [col for col in (CODE_COL, NAME_COL) if col not in df.columns]
    if missing:
        logger.error(f"市区町村マスタに必須列がありません: {missing}")
        raise KeyError(f"市区町村マスタに必須列がありません: {missing}")

    name_by_code = {
        int(code): str(name) for code, name in zip(df[CODE_COL], df[NAME_COL], strict=True)
    }
    logger.info(f"市区町村マスタを読み込みました: {len(name_by_code)} 件")
    return name_by_code


def code_to_label(code: int, name_by_code: dict[int, str]) -> str:
    """市区町村コードを表示用ラベルに変換する.

    マスタに名称があれば名称を、無ければコード文字列をフォールバックとして返す。

    Args:
        code: 市区町村コード。
        name_by_code: ``load_municipality_names`` が返す対応辞書。

    Returns:
        表示用ラベル（例: ``"千代田区"`` または未知コード時 ``"13999"``）。
    """
    return name_by_code.get(int(code), str(int(code)))
