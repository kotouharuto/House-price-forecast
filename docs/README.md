# ドキュメント索引

東京都不動産価格予測プロジェクトのドキュメント一覧。テーマ別にフォルダ分けしている。

## 構成

```
docs/
├── bi_tool/           BIツール本体の要件・フェーズ管理・実装サマリ
├── modeling/          モデル開発・実験管理・特徴量設計・評価指標
├── explainability/    説明可能性・AVM 査定（What-if、信頼区間、ロードマップ）
└── architecture/      システム全体設計・前処理規約
```

---

## `bi_tool/` — BIツール本体

予測結果可視化 BI ツールの要件定義からフェーズ管理、実装サマリまで。

| ファイル | 内容 |
|---|---|
| [spec.md](bi_tool/spec.md) | 要件定義書 |
| [phases.md](bi_tool/phases.md) | フェーズ別スコープ（P0〜P6） |
| [phases_flowchart.md](bi_tool/phases_flowchart.md) | フェーズ間のフローチャート |
| [p0_p1_improvement_plan.md](bi_tool/p0_p1_improvement_plan.md) | P0/P1 の改善計画 |
| [implementation_summary.md](bi_tool/implementation_summary.md) | 実装サマリ（最終版） |

---

## `modeling/` — モデル開発・実験

モデル実験のログ・提案・特徴量候補・評価指標の定義。

### 実験管理

| ファイル | 内容 |
|---|---|
| [experiment_log.md](modeling/experiment_log.md) | 実施済みの介入・スコア・観察 |
| [experiment_proposals.md](modeling/experiment_proposals.md) | 未実施の改善アイデア管理 |
| [experiment_git_workflow.md](modeling/experiment_git_workflow.md) | 実験時のブランチ運用・PR ルール |

### 評価・特徴量

| ファイル | 内容 |
|---|---|
| [evaluation_metrics.md](modeling/evaluation_metrics.md) | 評価指標の定義と読み方 |
| [correlation_feature_candidates.md](modeling/correlation_feature_candidates.md) | 相関分析にもとづく特徴量候補 |
| [external_feature_candidates.md](modeling/external_feature_candidates.md) | 外部データ由来の特徴量候補 |
| [prop_004_low_band_features.md](modeling/prop_004_low_band_features.md) | 低額帯（〜2000万）向けの特徴量強化詳細設計 |

---

## `explainability/` — 説明可能性・AVM 査定

説明可能性（What-if / 信頼区間 / 類似物件比較）と AVM 査定機能のロードマップ。

| ファイル | 内容 |
|---|---|
| [roadmap.md](explainability/roadmap.md) | Phase 1〜5 のロードマップ・成果物リンク |
| [whatif_analysis_guide.md](explainability/whatif_analysis_guide.md) | What-if 分析の限界と代替手法の整理 |
| [confidence_interval_guide.md](explainability/confidence_interval_guide.md) | 信頼区間（Quantile Regression）の読み方 |

---

## `architecture/` — システム設計・規約

パイプライン全体図と前処理時の規約。

| ファイル | 内容 |
|---|---|
| [system_pipeline_flowchart.md](architecture/system_pipeline_flowchart.md) | データ取得 〜 学習 〜 可視化までの全体パイプライン |
| [normalization_guidelines.md](architecture/normalization_guidelines.md) | カテゴリ変数の正規化ガイドライン |
