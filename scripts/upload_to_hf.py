"""学習済みモデルと予測結果ファイルを HuggingFace Hub にアップロードするスクリプト.

使い方:
    uv run python scripts/upload_to_hf.py

事前準備:
    huggingface-cli login  # HF_TOKEN を設定
"""

from pathlib import Path

from huggingface_hub import HfApi, create_repo

# プロジェクトルート
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# HuggingFace リポジトリ設定
_HF_REPO_ID = "Haruto0321/tokyo-rent-predictor-data"
_HF_REPO_TYPE = "dataset"

# アップロード対象ファイル: (ローカルパス, HF上のパス)
_UPLOAD_FILES: list[tuple[Path, str]] = [
    (_PROJECT_ROOT / "models" / "lgbm_model.pkl", "models/lgbm_model.pkl"),
    (_PROJECT_ROOT / "outputs" / "test_predictions.csv", "outputs/test_predictions.csv"),
    (
        _PROJECT_ROOT / "outputs" / "test_predictions_properties.geojson",
        "outputs/test_predictions_properties.geojson",
    ),
    (
        _PROJECT_ROOT / "outputs" / "test_predictions_stations.geojson",
        "outputs/test_predictions_stations.geojson",
    ),
]


def main() -> None:
    """アップロードを実行する."""
    api = HfApi()

    # リポジトリが存在しない場合は作成
    create_repo(repo_id=_HF_REPO_ID, repo_type=_HF_REPO_TYPE, exist_ok=True)
    print(f"リポジトリ確認済み: {_HF_REPO_ID}")

    for local_path, hf_path in _UPLOAD_FILES:
        if not local_path.exists():
            print(f"スキップ（ファイルなし）: {local_path}")
            continue

        print(f"アップロード中: {local_path.name} → {hf_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path,
            repo_id=_HF_REPO_ID,
            repo_type=_HF_REPO_TYPE,
        )
        print(f"完了: {hf_path}")

    print("\n全ファイルのアップロードが完了しました。")
    print(f"確認URL: https://huggingface.co/datasets/{_HF_REPO_ID}")


if __name__ == "__main__":
    main()
