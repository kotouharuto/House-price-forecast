"""配信アセット（モデル・データ）の取得を一元管理するモジュール.

Streamlit Cloud では学習済みモデルや特徴量データを Git に含めず、HuggingFace
Hub の dataset リポジトリに置いて起動時にダウンロードする。アップロード
（``scripts/upload_to_hf.py``）とダウンロード（各 Streamlit ページ）で
ファイル一覧が二重管理にならないよう、``ASSETS`` を唯一の真実とする。
"""

import shutil
import sys
from pathlib import Path

import streamlit as st

# プロジェクトルートを sys.path に追加（app.xxx / src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# HuggingFace リポジトリ設定
HF_REPO_ID = "Haruto0321/tokyo-rent-predictor-data"
HF_REPO_TYPE = "dataset"

# 配信アセットの一覧: (HF 上のパス, ローカルパス)
# upload 側はこれを (local, hf) に並べ替えて使う。
ASSETS: list[tuple[str, Path]] = [
    # 点予測モデル
    ("models/lgbm_model.pkl", _PROJECT_ROOT / "models" / "lgbm_model.pkl"),
    # 分位点回帰モデル（物件査定ページの予測区間に使用）
    ("models/lgbm_quantile_low.pkl", _PROJECT_ROOT / "models" / "lgbm_quantile_low.pkl"),
    ("models/lgbm_quantile_med.pkl", _PROJECT_ROOT / "models" / "lgbm_quantile_med.pkl"),
    ("models/lgbm_quantile_high.pkl", _PROJECT_ROOT / "models" / "lgbm_quantile_high.pkl"),
    # 予測結果（BI / 地図ページ）
    ("outputs/test_predictions.csv", _PROJECT_ROOT / "outputs" / "test_predictions.csv"),
    (
        "outputs/test_predictions_properties.geojson",
        _PROJECT_ROOT / "outputs" / "test_predictions_properties.geojson",
    ),
    (
        "outputs/test_predictions_stations.geojson",
        _PROJECT_ROOT / "outputs" / "test_predictions_stations.geojson",
    ),
    # 特徴量データ（物件査定ページの類似物件検索・予測に使用）
    ("data/processed/features.csv", _PROJECT_ROOT / "data" / "processed" / "features.csv"),
]


def _get_hf_token() -> str | None:
    """Streamlit Secrets から HF_TOKEN を取得する（無ければ None）.

    ローカル開発環境では ``.streamlit/secrets.toml`` が無い・無設定でも動くよう、
    例外を握りつぶして匿名アクセス（``token=None``）にフォールバックする。
    """
    try:
        return st.secrets["HF_TOKEN"]
    except (FileNotFoundError, KeyError, AttributeError):
        return None


@st.cache_resource(show_spinner="データを準備しています...")
def download_assets() -> None:
    """HuggingFace Hub から必要なファイルをダウンロードする.

    Private リポジトリの場合は Streamlit Secrets の ``HF_TOKEN`` を利用する。
    既にローカルに存在する場合はスキップする（ローカル開発環境への配慮）。
    ``@st.cache_resource`` によりセッション/コンテナごとに1回だけ実行されるため、
    複数ページから呼び出しても多重ダウンロードは発生しない。
    """
    from huggingface_hub import hf_hub_download

    token = _get_hf_token()

    for hf_path, local_path in ASSETS:
        if local_path.exists():
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            repo_type=HF_REPO_TYPE,
            local_dir=str(_PROJECT_ROOT),
            token=token,
        )
        # hf_hub_download はキャッシュディレクトリに置くため、指定パスへコピー
        if Path(downloaded) != local_path:
            shutil.copy2(downloaded, local_path)
