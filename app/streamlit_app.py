"""賃料予測アプリのエントリポイント.

Streamlit Cloud 起動時に HuggingFace Hub から必要なファイルを自動ダウンロードする。
ローカル環境では既にファイルが存在する場合はスキップする。
"""

import sys
from pathlib import Path

import streamlit as st

# プロジェクトルートを sys.path に追加
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# HuggingFace リポジトリ設定
_HF_REPO_ID = "Haruto0321/tokyo-rent-predictor-data"

# ダウンロード対象: (HF上のパス, ローカル保存先)
_DOWNLOAD_FILES: list[tuple[str, Path]] = [
    ("models/lgbm_model.pkl", _PROJECT_ROOT / "models" / "lgbm_model.pkl"),
    ("outputs/test_predictions.csv", _PROJECT_ROOT / "outputs" / "test_predictions.csv"),
    (
        "outputs/test_predictions_properties.geojson",
        _PROJECT_ROOT / "outputs" / "test_predictions_properties.geojson",
    ),
    (
        "outputs/test_predictions_stations.geojson",
        _PROJECT_ROOT / "outputs" / "test_predictions_stations.geojson",
    ),
]


@st.cache_resource(show_spinner="データを準備しています...")
def _download_assets() -> None:
    """HuggingFace Hub から必要なファイルをダウンロードする.

    既にローカルに存在する場合はスキップする（ローカル開発環境への配慮）。
    """
    import shutil

    from huggingface_hub import hf_hub_download

    for hf_path, local_path in _DOWNLOAD_FILES:
        if local_path.exists():
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            local_dir=str(_PROJECT_ROOT),
        )
        # hf_hub_download はキャッシュディレクトリに置くため、指定パスにコピー
        if Path(downloaded) != local_path:
            shutil.copy2(downloaded, local_path)


def main() -> None:
    """エントリポイント."""
    _download_assets()

    st.set_page_config(
        page_title="東京賃料予測",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 東京都賃料予測 AVM")
    st.markdown("サイドバーから各ページを選択してください。")


if __name__ == "__main__":
    main()
