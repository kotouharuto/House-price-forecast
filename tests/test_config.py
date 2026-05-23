"""設定ローダ（src.utils.config）のテスト."""

from pathlib import Path

import pytest

from src.utils.config import load_model_params

# テスト用YAMLの内容（正常系）
_VALID_YAML = """\
lgbm:
  objective: regression
  n_estimators: 1000
  learning_rate: 0.05
  random_state: 42
xgb:
  max_depth: 6
"""


def _write_yaml(tmp_path: Path, content: str) -> Path:
    """一時ディレクトリにYAMLを書き出し、そのパスを返すヘルパー."""
    params_path = tmp_path / "model_params.yaml"
    params_path.write_text(content, encoding="utf-8")
    return params_path


def test_load_model_params_returns_section_dict(tmp_path: Path) -> None:
    """指定セクションの内容が型を保ったまま辞書で返ること."""
    params_path = _write_yaml(tmp_path, _VALID_YAML)

    params = load_model_params("lgbm", params_path)

    assert params == {
        "objective": "regression",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "random_state": 42,
    }
    # YAMLの型推論が意図通りであること（int / float / str）
    assert isinstance(params["n_estimators"], int)
    assert isinstance(params["learning_rate"], float)
    assert isinstance(params["objective"], str)


def test_load_model_params_selects_requested_section(tmp_path: Path) -> None:
    """セクション指定で対象セクションのみが返ること."""
    params_path = _write_yaml(tmp_path, _VALID_YAML)

    params = load_model_params("xgb", params_path)

    assert params == {"max_depth": 6}


def test_load_model_params_raises_when_file_missing(tmp_path: Path) -> None:
    """ファイルが存在しない場合に FileNotFoundError を送出すること."""
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        load_model_params("lgbm", missing_path)


def test_load_model_params_raises_when_section_missing(tmp_path: Path) -> None:
    """指定セクションが存在しない場合に KeyError を送出すること."""
    params_path = _write_yaml(tmp_path, _VALID_YAML)

    with pytest.raises(KeyError):
        load_model_params("not_exist", params_path)


def test_load_model_params_raises_when_root_not_mapping(tmp_path: Path) -> None:
    """YAML全体が辞書でない場合に TypeError を送出すること."""
    params_path = _write_yaml(tmp_path, "- just\n- a\n- list\n")

    with pytest.raises(TypeError):
        load_model_params("lgbm", params_path)


def test_load_model_params_raises_when_section_not_mapping(tmp_path: Path) -> None:
    """セクション直下がマッピングでない場合に TypeError を送出すること."""
    params_path = _write_yaml(tmp_path, "lgbm: 42\n")

    with pytest.raises(TypeError):
        load_model_params("lgbm", params_path)
