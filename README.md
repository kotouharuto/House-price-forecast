# 🏠 Tokyo Rent Predictor

> 国土交通省「不動産情報ライブラリ」のデータを活用した東京都賃貸物件の賃料予測モデル

## 📋 概要
国土交通省「不動産情報ライブラリ」から取得した東京都の不動産データをもとに、
機械学習モデル（LightGBM / XGBoost）で賃料を予測するEnd-to-Endプロジェクトです。
予測結果はStreamlitアプリとしてインタラクティブに操作できます。

## 🎯 プロジェクトの目的
- 物件の属性情報（面積・築年数・駅徒歩・エリア等）から賃料を予測する
- SHAPによる予測要因の可視化で「なぜこの賃料になるか」を説明可能にする
- 公式オープンデータを起点とするEnd-to-Endパイプラインを構築する

## 🏗️ アーキテクチャ
（ここに後でMermaidのフロー図を追加予定）

データ取得 → 前処理 → EDA → 特徴量設計 → モデリング → 評価 → アプリ化

## 📊 使用データ
- **ソース**: 国土交通省「不動産情報ライブラリ」（https://www.reinfolib.mlit.go.jp/）
- **対象エリア**: 東京都23区
- **取得項目**: 賃料、面積、築年数、最寄駅、駅徒歩、階数、構造、間取り、所在地 等
- **ライセンス**: 政府標準利用規約（オープンデータ）

## 🛠️ 技術スタック
| カテゴリ | 技術 |
|---------|------|
| 言語 | Python 3.12 |
| 環境管理 | uv |
| データ処理 | pandas, NumPy |
| 機械学習 | LightGBM, XGBoost, scikit-learn |
| 説明可能性 | SHAP |
| 可視化 | matplotlib, seaborn, Plotly |
| アプリ | Streamlit |
| コード品質 | Ruff, mypy, pytest |

## 📁 ディレクトリ構成
```
tokyo-rent-predictor/
├── README.md
├── pyproject.toml
├── .python-version          # 3.12
├── .gitignore
├── CLAUDE.md                # Claude Code用の開発ガイドライン
├── src/
│   ├── preprocessing/
│   │   └── clean.py            # データ前処理・特徴量エンジニアリング
│   ├── modeling/
│   │   ├── train.py            # モデル学習
│   │   └── evaluate.py         # モデル評価・SHAP可視化
│   └── utils/
│       ├── config.py           # 設定値管理
│       └── logger.py           # ロガー設定
├── notebooks/
│   ├── 01_eda.ipynb            # 探索的データ分析
│   ├── 02_feature_engineering.ipynb  # 特徴量設計の検討
│   └── 03_modeling.ipynb       # モデリング実験
├── app/
│   └── streamlit_app.py        # Streamlitアプリ
├── data/
│   ├── raw/                    # 取得済みの生データ（.gitignore対象）
│   └── processed/              # 前処理済みデータ（.gitignore対象）
├── models/                     # 学習済みモデル（.gitignore対象）
├── outputs/
│   └── figures/                # EDA・SHAP等の可視化画像
└── tests/
    └── test_preprocessing.py   # 前処理のテスト
```

## 🚀 セットアップ

### 前提条件
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)（Rust製の高速なPythonパッケージマネージャ）

### 1. uv のインストール
```bash
# macOS（Homebrew）
brew install uv

# 公式インストーラ（macOS / Linux）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows（PowerShell）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール確認：
```bash
uv --version
```

### 2. リポジトリのクローン
```bash
git clone https://github.com/kotouharuto/Rental-price-forecast.git
cd Rental-price-forecast
```

### 3. 仮想環境 + 依存関係のセットアップ
uv が自動で `.venv/` を作成し、`pyproject.toml` / `uv.lock` から依存をインストールします。

```bash
# 本番依存のみ
uv sync

# 開発用依存（Ruff / pytest / mypy）も含める ← 通常はこちら
uv sync --group dev
```

> **Tip:** `.venv` が壊れた場合や依存解決が怪しい場合は、以下で作り直すのが確実です。
> ```bash
> rm -rf .venv
> uv sync --group dev
> ```

### 4. データ配置
不動産情報ライブラリから取得した CSV を `data/raw/` 配下に配置してください
（データ自体は `.gitignore` 対象）。

---

## 🧪 開発コマンド

### Lint / Format（Ruff）
```bash
# Lint チェック（修正なし、問題の検出のみ）
uv run ruff check .

# Lint を自動修正
uv run ruff check . --fix

# さらに踏み込んだ修正（未使用import削除など、安全でない変更も適用）
uv run ruff check . --fix --unsafe-fixes

# フォーマット差分の確認（CI向け）
uv run ruff format --check .

# フォーマットを適用
uv run ruff format .
```

開発の基本フロー：
```bash
uv run ruff format .          # 1. 整形
uv run ruff check . --fix     # 2. Lint修正
uv run mypy src/              # 3. 型チェック
uv run pytest                 # 4. テスト
```

### 型チェック（mypy）
```bash
uv run mypy src/
```

### テスト（pytest）
```bash
# 全テスト実行
uv run pytest

# 詳細出力（失敗時のtraceback付き）
uv run pytest -v

# 特定ファイルのみ
uv run pytest tests/test_preprocessing.py

# カバレッジ計測（pytest-cov が入っている場合）
uv run pytest --cov=src --cov-report=term-missing
```

### Streamlit アプリ起動
```bash
uv run streamlit run app/streamlit_app.py
```

### 依存関係の追加・削除
```bash
# 本番依存を追加
uv add pandas

# 開発用依存を追加
uv add --dev pytest-cov

# 依存を削除
uv remove <パッケージ名>

# 依存をロックファイル(uv.lock)に従って完全再現
uv sync --frozen
```

### よく使うコマンド早見表
| コマンド | 用途 |
|---|---|
| `uv sync --group dev` | 依存関係（開発用含む）のインストール・更新 |
| `uv add <pkg>` / `uv add --dev <pkg>` | 依存パッケージを追加 |
| `uv run ruff check . --fix` | Lint＆自動修正 |
| `uv run ruff format .` | コードフォーマット適用 |
| `uv run mypy src/` | 型チェック |
| `uv run pytest` | テスト実行 |
| `uv run streamlit run app/streamlit_app.py` | アプリ起動 |

## 📈 分析結果
（プロジェクト完成後に追記）
- モデル精度（RMSE, MAE, R²）
- SHAP要因分解の可視化
- エリアごとの賃料分布

## 🖥️ デモ
（Streamlit Cloudへデプロイ後にURLを追記）

## 📝 関連記事
（Zenn記事を書いた後にリンクを追記）

## ライセンス
MIT License
