"""設定ローダ（src.utils.config）のテスト."""

from pathlib import Path

import pytest

from src.utils.config import load_model_params

# テスト用YAML（必須セクション lgbm / split を含む）
_VALID_YAML = """\
lgbm:
  objective: regression
  n_estimators: 1000
  learning_rate: 0.05
  random_state: 42
split:
  test_size: 0.25
  random_state: 42
"""


def _write_yaml(tmp_path: Path, content: str) -> Path:
    """一時ディレクトリにYAMLを書き出し、そのパスを返すヘルパー."""
    params_path = tmp_path / "model_params.yaml"
    params_path.write_text(content, encoding="utf-8")
    return params_path


def test_load_model_params_returns_dict_with_required_sections(tmp_path: Path) -> None:
    """必須セクション (lgbm / split) を含む辞書が返ること."""
    params_path = _write_yaml(tmp_path, _VALID_YAML)

    params = load_model_params(params_path)

    assert {"lgbm", "split"}.issubset(params.keys())
    assert params["lgbm"]["n_estimators"] == 1000
    # YAMLの型推論が意図通りであること（int / float / str）
    assert isinstance(params["lgbm"]["n_estimators"], int)
    assert isinstance(params["lgbm"]["learning_rate"], float)
    assert isinstance(params["lgbm"]["objective"], str)
    assert params["split"]["test_size"] == 0.25


def test_load_model_params_uses_default_path_when_not_specified() -> None:
    """引数なしで既定パス（configs/model_params.yaml）を読み込めること."""
    params = load_model_params()

    assert {"lgbm", "split"}.issubset(params.keys())


def test_load_model_params_raises_when_file_missing(tmp_path: Path) -> None:
    """ファイルが存在しない場合に FileNotFoundError を送出すること."""
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        load_model_params(missing_path)


def test_load_model_params_raises_when_required_section_missing(tmp_path: Path) -> None:
    """必須セクションが欠落している場合に KeyError を送出すること."""
    incomplete_yaml = "lgbm:\n  n_estimators: 100\n"  # split が無い
    params_path = _write_yaml(tmp_path, incomplete_yaml)

    with pytest.raises(KeyError, match="split"):
        load_model_params(params_path)


def test_load_model_params_includes_missing_section_name_in_error(tmp_path: Path) -> None:
    """エラーメッセージに不足セクション名が含まれること."""
    params_path = _write_yaml(tmp_path, "lgbm:\n  n_estimators: 100\n")

    with pytest.raises(KeyError) as exc_info:
        load_model_params(params_path)
    assert "split" in str(exc_info.value)
