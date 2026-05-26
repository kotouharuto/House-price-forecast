"""地理マスタ（src.visualization.master）のテスト."""

from pathlib import Path

import pytest

from src.visualization.master import (
    code_to_label,
    load_municipality_names,
)

# テスト用マスタCSVの内容
_MASTER_CSV = "市区町村コード,市区町村名\n13101,千代田区\n13123,江戸川区\n"


def _write_master(tmp_path: Path, content: str) -> Path:
    """一時ディレクトリにマスタCSVを書き出してパスを返す."""
    path = tmp_path / "master.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_municipality_names_returns_code_to_name(tmp_path: Path) -> None:
    """コード(int)→名称(str) の辞書を返すこと."""
    path = _write_master(tmp_path, _MASTER_CSV)

    name_by_code = load_municipality_names(path)

    assert name_by_code == {13101: "千代田区", 13123: "江戸川区"}
    assert isinstance(next(iter(name_by_code)), int)


def test_load_municipality_names_raises_when_file_missing(tmp_path: Path) -> None:
    """マスタCSVが存在しない場合に FileNotFoundError を送出すること."""
    with pytest.raises(FileNotFoundError):
        load_municipality_names(tmp_path / "missing.csv")


def test_load_municipality_names_raises_when_column_missing(tmp_path: Path) -> None:
    """必須列が欠落している場合に KeyError を送出すること."""
    path = _write_master(tmp_path, "市区町村コード\n13101\n")

    with pytest.raises(KeyError):
        load_municipality_names(path)


def test_code_to_label_uses_name_when_known() -> None:
    """既知コードは名称を返すこと."""
    assert code_to_label(13101, {13101: "千代田区"}) == "千代田区"


def test_code_to_label_falls_back_to_code_when_unknown() -> None:
    """未知コードはコード文字列をフォールバックとして返すこと."""
    assert code_to_label(13999, {13101: "千代田区"}) == "13999"


def test_real_master_covers_known_wards() -> None:
    """同梱マスタが代表的な区を正しく対応付けていること（回帰防止）."""
    name_by_code = load_municipality_names()

    assert name_by_code[13101] == "千代田区"
    assert name_by_code[13103] == "港区"
    assert name_by_code[13113] == "渋谷区"
    assert name_by_code[13213] == "東村山市"
