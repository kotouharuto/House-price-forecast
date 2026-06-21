"""配信アセットの共有マニフェスト（app.assets）のテスト.

Streamlit ランタイム外でも import 可能な範囲（``ASSETS`` の整合性）を検証する。
``download_assets`` 自体はネットワーク/Streamlit ランタイムに依存するため対象外。
"""

from pathlib import Path

from app.assets import ASSETS, HF_REPO_ID, HF_REPO_TYPE

# 物件査定ページ（Phase 4）が依存するアセットの HF パス
_REQUIRED_FOR_APPRAISAL = {
    "models/lgbm_model.pkl",
    "models/lgbm_quantile_low.pkl",
    "models/lgbm_quantile_med.pkl",
    "models/lgbm_quantile_high.pkl",
    "data/processed/features.csv",
}


def test_repo_config_is_dataset() -> None:
    """リポジトリ設定が dataset 型であること."""
    assert HF_REPO_TYPE == "dataset"
    assert HF_REPO_ID


def test_assets_cover_appraisal_dependencies() -> None:
    """査定ページが必要とするファイルがすべてマニフェストに含まれること."""
    hf_paths = {hf_path for hf_path, _ in ASSETS}
    missing = _REQUIRED_FOR_APPRAISAL - hf_paths
    assert not missing, f"マニフェストに不足しています: {missing}"


def test_assets_have_no_duplicate_hf_paths() -> None:
    """HF パスの重複が無いこと."""
    hf_paths = [hf_path for hf_path, _ in ASSETS]
    assert len(hf_paths) == len(set(hf_paths))


def test_local_path_matches_hf_path() -> None:
    """各エントリのローカルパスが HF パスと同じ相対構造で終わること.

    upload と download で同じファイルが対応するよう、末尾の相対パスを揃える。
    """
    for hf_path, local_path in ASSETS:
        assert isinstance(local_path, Path)
        assert local_path.as_posix().endswith(hf_path), (
            f"パス不一致: hf={hf_path} / local={local_path}"
        )
