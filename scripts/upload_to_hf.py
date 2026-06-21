"""学習済みモデルと予測結果ファイルを HuggingFace Hub にアップロードするスクリプト.

使い方:
    uv run python scripts/upload_to_hf.py

事前準備:
    huggingface-cli login  # HF_TOKEN を設定
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# プロジェクトルートを sys.path に追加（app.assets を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.assets import ASSETS, HF_REPO_ID, HF_REPO_TYPE  # noqa: E402

# アップロード対象: 共有マニフェスト（HF上パス, ローカルパス）を (ローカルパス, HF上パス) に並べ替える
_UPLOAD_FILES: list[tuple[Path, str]] = [(local_path, hf_path) for hf_path, local_path in ASSETS]


def main() -> None:
    """アップロードを実行する."""
    api = HfApi()

    # リポジトリが存在しない場合は作成
    create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, exist_ok=True)
    print(f"リポジトリ確認済み: {HF_REPO_ID}")

    for local_path, hf_path in _UPLOAD_FILES:
        if not local_path.exists():
            print(f"スキップ（ファイルなし）: {local_path}")
            continue

        print(f"アップロード中: {local_path.name} → {hf_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )
        print(f"完了: {hf_path}")

    print("\n全ファイルのアップロードが完了しました。")
    print(f"確認URL: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
