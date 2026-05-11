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
- uv

### インストール
```bash
# リポジトリをクローン
git clone https://github.com/kotouharuto/Rental-price-forecast.git
cd Rental-price-forecast

# uv で依存関係をインストール
uv sync

# 開発用依存関係を含める場合
uv sync --extra dev
```

### データ配置
不動産情報ライブラリから取得した CSV/JSON を `data/raw/` 配下に配置してください
（データ自体は `.gitignore` 対象です）。

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
