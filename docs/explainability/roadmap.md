# 説明可能性・AVM 査定 ロードマップ

不動産 AVM（自動価格査定）向けに、**顧客説明力**と **予測信頼度** を提供する機能群の開発計画。
What-if 分析が AVM 文脈で限定的だった反省を踏まえ、業界実務に整合する「**類似物件比較**」と「**信頼区間**」を中心に据える。

- 思想の整理: [whatif_analysis_guide.md](whatif_analysis_guide.md)（What-if の限界と代替手法）
- BI ツールの全体像: [../bi_tool/phases.md](../bi_tool/phases.md)

> **状態の凡例**: ✅ 完了 / 🔄 進行中 / ⚠️ 一部未達 / ⬜ 未着手

---

## フェーズ一覧

| フェーズ | 内容 | 状態 |
|---|---|---|
| **Phase 1: 類似物件比較（関数）** | 階層フィルタ + 正規化ユークリッド距離による kNN | ✅ 完了 |
| **Phase 2: Quantile Regression（信頼区間）** | LightGBM 分位回帰で予測区間を出す | ✅ 完了 |
| **Phase 3: 類似物件からの実証的区間** | 類似物件 N 件の実取引価格分位点から区間を出す | ⬜ 未着手 |
| **Phase 4: Streamlit ページ統合** | 顧客向け査定 UI（類似物件 + 区間の表示） | ⬜ 未着手 |
| **Phase 5: モデル評価への組込** | 信頼区間カバレッジ率の評価指標化 | ⬜ 未着手 |

---

## Phase 1: 類似物件比較（関数） ✅

不動産鑑定の「取引事例比較法」に相当する説明手法を、純粋関数として提供する。

### スコープ
- 階層フィルタ（種類・市区町村コード）で母集団を絞り、面積・築年数・最寄駅距離の正規化ユークリッド距離で kNN
- 対象物件は除外、欠損のある行は距離計算から除外
- `recent_first=True` で取引年・取引四半期降順に並び替え

### タスク
- ✅ `src/preprocessing/feature_engineering.py` で `取引年` を drop しないよう変更
- ✅ `src/modeling/train.py` で `取引年` を学習特徴量から除外（`_NON_FEATURE_COLUMNS`）
- ✅ `src/visualization/similar_properties.py`: `find_similar_properties()` を実装
- ✅ `tests/test_similar_properties.py`: ユニットテスト 11 件追加
- ✅ ruff / pytest 通過確認（pytest 78 件全 PASS）
- ✅ `notebooks/05_property_appraisal.ipynb` で動作確認（ユーザー作業中）

### 成果物
- [src/visualization/similar_properties.py](../src/visualization/similar_properties.py)
- [tests/test_similar_properties.py](../tests/test_similar_properties.py)

---

## Phase 2: Quantile Regression（信頼区間） ✅

LightGBM の `objective="quantile"` を使い、点推定だけでなく**統計的な予測区間**を返せるようにする。

### スコープ
- 分位点 α = 0.05 / 0.50 / 0.95 の 3 モデルを学習
- 既存の `lgbm_model.pkl` は触らない（補助モデル扱い）
- 予測ユーティリティ関数 `predict_with_interval()` を追加
- カバレッジ率（実価格が予測区間に入る割合）の検証

### タスク
- ✅ `configs/model_params.yaml` に `quantile:` セクション追加（α リスト・流用ハイパラ）
- ✅ `src/modeling/train_quantile.py`: 3 モデル学習スクリプト新規
- ✅ モデル保存: `models/lgbm_quantile_low.pkl` / `..._med.pkl` / `..._high.pkl`
- ✅ `src/visualization/prediction.py`: `predict_with_interval()` 関数（lower / median / upper を返す）
- ✅ `tests/test_train_quantile.py` / `tests/test_prediction.py`: 単体テスト 12 件追加
- ✅ テストセットでカバレッジ率を計測（学習スクリプト末尾でログ出力）
- ✅ ruff / pytest 通過確認（pytest 90 件全 PASS）

### 設計判断（採用）
- 既存 `lgbm_model.pkl` を **置き換えない**（BI ツール影響をゼロに）
- `train_quantile.py` で **3 モデル別学習**（α ごとに独立）
- 学習データは既存 `prepare_dataset()` を流用
- YAML は `lgbm` セクションを継承し `quantile.overrides` で objective / metric のみ上書き

### 成果物
- [src/modeling/train_quantile.py](../src/modeling/train_quantile.py)
- [src/visualization/prediction.py](../src/visualization/prediction.py)
- [tests/test_train_quantile.py](../tests/test_train_quantile.py)
- [tests/test_prediction.py](../tests/test_prediction.py)
- `models/lgbm_quantile_{low,med,high}.pkl`

### 初回学習結果（2026-06-14）
| 項目 | 値 | 評価 |
|---|---|---|
| 名目カバレッジ率 | 90% | — |
| **実カバレッジ率** | **78.94%** | ⚠️ 目標 85〜95% を下回る |
| 差分 | −11.06pp | 区間が**狭すぎる** = モデル過信 |
| 区間幅 中央値 | 2,007 万円 | — |
| 区間幅 平均 | 3,402 万円 | — |

⚠️ **キャリブレーション要改善**。実カバレッジが目標を下回っているため、Phase 5（評価指標化）で価格帯別の分析を行い、α 拡張やハイパラ調整を検討する。Phase 2 のスコープ（実装と検証パイプライン整備）は完了。

---

## Phase 3: 類似物件からの実証的区間 ⬜

過去取引の**事実データ**から、対象物件の予測区間を統計的に出す。Phase 1 の関数を再利用するだけで概ね実装可能。

### スコープ
- 類似物件 N 件（例: 30 件）の実取引価格の分位点を区間として返す
- Phase 2 の Quantile Regression 区間と並べて表示できるようにする
- 両区間の乖離が大きいときは「モデル不確実」フラグを立てる

### タスク
- ⬜ `src/visualization/prediction.py` に `empirical_interval_from_similar()` 関数追加
- ⬜ パラメータ: `n_similar`（既定 30）、`lower_q` / `upper_q`（既定 0.10 / 0.90）
- ⬜ Quantile 区間と実証的区間を並べて返す統合関数 `compare_intervals()`
- ⬜ テスト追加（小規模 DataFrame で分位点が想定通り出ること）
- ⬜ ノートブック `05_property_appraisal.ipynb` で可視化プロトタイプ

---

## Phase 4: Streamlit ページ統合 ⬜

顧客向け査定書のレイアウトを Streamlit 上で提供する。**広い意味での production 化**。

### スコープ
- 対象物件を選択（市区町村 / 種類 / 価格帯 / 個別選択）
- 予測値（点 + 区間）+ 類似物件 5 件 + 実取引価格分布を 1 画面で表示
- 査定の信頼度判定ロジック（区間幅・乖離・類似物件数で 3 段階表示）

### タスク
- ⬜ `app/pages/4_物件査定.py` 新規作成
- ⬜ 物件選択 UI（サイドバーで条件絞り込み → セレクトボックス）
- ⬜ 予測値カード（中央値 + 区間バー）
- ⬜ 類似物件テーブル（[../bi_tool/phases.md](../bi_tool/phases.md) の表示規約に準拠）
- ⬜ 類似物件の実価格分布ヒストグラム
- ⬜ 査定信頼度の判定（高 / 中 / 低）
- ⬜ `test_predictions.csv` への `取引年` 追加（Phase 4 で恒久対応する想定）
- ⬜ `home.py` の導線にリンク追加

### 依存
- Phase 2 完了（予測区間が必要）
- Phase 3 完了（実証的区間が必要）

---

## Phase 5: モデル評価への組込 ⬜

予測の点精度だけでなく、**区間予測の質**を評価指標に加える。

### スコープ
- 予測区間カバレッジ率（PICP: Prediction Interval Coverage Probability）
- 予測区間幅の中央値（PIAW: Prediction Interval Average Width）
- 価格帯別のカバレッジ・幅
- BI ダッシュボード（`app/pages/1_BIダッシュボード.py`）に追加

### タスク
- ⬜ `src/visualization/aggregate.py` に区間評価関数追加（`coverage_rate()` / `interval_width_stats()`）
- ⬜ 評価指標一覧ページ（`app/pages/3_評価指標一覧.py`）に PICP / PIAW の解説追加
- ⬜ BI ダッシュボードに「区間内率」KPI を追加
- ⬜ 区間が極端に広い物件 = 学習データ分布外 のフラグ付け
- ⬜ テスト追加

---

## 想定外スコープ（やらない・あとで）

| 項目 | やらない理由 |
|---|---|
| What-if 分析の顧客向け公開 | 不動産 AVM 文脈で因果と誤読される恐れが大きい。社内 QA ツール限定とする |
| Conformal Prediction | Quantile Regression で十分な精度を確保できる前提。要件が出てから検討 |
| Leaf Embedding 類似度 | Phase 1 の階層フィルタで実用十分。性能不足が出てから検討 |
| Counterfactual Explanation（DiCE 等） | 不動産の特徴量は変えられないものが大半で実務的意義が薄い |

---

## 関連文書

- [whatif_analysis_guide.md](whatif_analysis_guide.md) — What-if 分析の限界と代替手法の整理
- [../bi_tool/phases.md](../bi_tool/phases.md) — BI ツール本体のフェーズ管理
- [../modeling/evaluation_metrics.md](../modeling/evaluation_metrics.md) — 既存評価指標の定義
- [../modeling/experiment_log.md](../modeling/experiment_log.md) — モデル実験ログ
