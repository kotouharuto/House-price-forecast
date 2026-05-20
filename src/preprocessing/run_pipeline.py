"""前処理・特徴量エンジニアリングのパイプラインを実行するスクリプト.

生データを読み込み、前処理（clean）→ 特徴量エンジニアリング（FE）を施して、
特徴量付きデータを ``data/processed/`` に保存する動作確認用スクリプト。

実行方法:
    uv run python -m src.preprocessing.run_pipeline
"""

# 標準ライブラリ
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（src.xxx を import するため）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# プロジェクト内モジュール（sys.path 操作後である必要があるため E402 を許容）
from src.preprocessing.clean import preprocess_data  # noqa: E402
from src.preprocessing.feature_engineering import engineer_features  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# 入出力パス（プロジェクトルート基準の絶対パスで固定）
_RAW_DATA_PATH = _PROJECT_ROOT / "data" / "raw" / "Tokyo_20251_20254.csv"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "features.csv"


def run_pipeline(raw_path: Path = _RAW_DATA_PATH, output_path: Path = _OUTPUT_PATH) -> None:
    """前処理 → 特徴量エンジニアリングを実行し、結果をCSVに保存する.

    Args:
        raw_path: 生データCSVのパス。
        output_path: 特徴量付きデータの出力先パス。
    """
    logger.info("=== パイプライン開始 ===")

    # 1. 前処理（欠損補完・和暦変換・外れ値処理・重複削除）
    df = preprocess_data(str(raw_path))
    logger.info(f"前処理完了: shape={df.shape}")

    # 2. 特徴量エンジニアリング（駅情報結合・カテゴリ化・派生特徴量生成）
    df = engineer_features(df)
    logger.info(f"特徴量エンジニアリング完了: shape={df.shape}")

    # 3. 特徴量付きデータを保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"特徴量付きデータを保存: {output_path}")

    logger.info("=== パイプライン完了 ===")

    # 動作確認用のサマリをコンソールに出力
    print("\n=== パイプライン実行結果 ===")
    print(f"出力ファイル: {output_path}")
    print(f"行数 x 列数: {df.shape}")
    print("\n--- 列名と型 ---")
    print(df.dtypes.to_string())
    print("\n--- 欠損数（上位10列） ---")
    print(df.isna().sum().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    run_pipeline()
