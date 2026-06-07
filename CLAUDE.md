# CLAUDE.md

## プロジェクト概要
東京都賃貸物件の賃料予測プロジェクト。国土交通省「不動産情報ライブラリ」から取得したデータをもとに、前処理 → ML → Streamlitアプリの一気通貫構成。

## 技術スタック
- Python 3.12 / uv
- データソース: 国土交通省「不動産情報ライブラリ」（オープンデータ）
- ML: LightGBM, XGBoost, scikit-learn
- 可視化: SHAP, matplotlib, seaborn, Plotly
- アプリ: Streamlit

## 開発ルール
- コードフォーマット: Ruff（`uv run ruff check .` / `uv run ruff format .`）
- 型チェック: mypy（`uv run mypy src/`）
- テスト: pytest（`uv run pytest`）
- コミットメッセージ: Conventional Commits（feat:, fix:, docs:, refactor:）
- docstring: Google style

## ディレクトリルール
- `src/`: プロダクションコード
- `notebooks/`: 探索・実験用（番号付き）
- `app/`: Streamlitアプリ
- `data/raw/`: 取得済みの生データ（Git管理外）
- `data/processed/`: 前処理済み（Git管理外）
- `models/`: 学習済みモデル（Git管理外）
- `outputs/figures/`: 可視化画像

## よく使うコマンド
- `uv run streamlit run app/home.py` - アプリ起動
- `uv run pytest` - テスト
- `uv run ruff check . --fix` - Lint修正

## ロギング
- `print` ではなく `src/utils/logger.py` の `get_logger` を使う
- ログは `logs/app.log` にのみ出力（コンソールには出さない方針）
- 5MB × 3世代でローテーション

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("学習開始: model=LightGBM")
logger.warning("欠損値が想定より多い")
logger.error("学習失敗", exc_info=True)
```
