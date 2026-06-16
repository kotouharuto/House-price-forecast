# モデル実験における Git / PR 運用ルール

ML 実験は性質上「不採用」で終わる割合が高いため、ナイーブに全PRをマージすると
`main` ブランチが捨てられた実験コードや巨大な生成物で汚染されていく。
本ドキュメントでは、本プロジェクトで採用する Git / PR 運用ルールを定義する。

関連: [experiment_log.md](experiment_log.md) /
[experiment_proposals.md](experiment_proposals.md)

---

## 1. 基本原則

実験ブランチで発生する変更を3つの軸で分類し、それぞれ独立した扱いをする。

| 軸 | 採用実験（採用条件を満たす） | 不採用実験 |
|---|---|---|
| **コード変更** (`src/`, `configs/`) | `main` にマージ | `main` にマージしない |
| **記録ドキュメント** (`docs/model_experiment_*.md`) | `main` にマージ | **`main` にマージする**（学びを残す） |
| **生成物** (`outputs/*.csv`, `models/best_params.yaml`, `models/*.pkl`) | 原則 `.gitignore` | 同左 |

要点:
- **コードは採否で扱いが変わる**: 採用された改善のみが `main` に積まれる
- **ドキュメントは常に残す**: 不採用実験も「過去に試した」「なぜ採らなかった」を記録
- **生成物は常に除外**: 再生成可能なため Git 管理外

---

## 2. ブランチ運用

### 命名規則
- 実験ブランチ: `exp-XXX` または `exp-XXX-<短い説明>`（例: `exp-007-clip-5oku`）
- ドキュメント単独ブランチ（必要時）: `docs-exp-XXX`

### ブランチの寿命
- 不採用実験のブランチも**削除しない**: 後で「あの実験はどうやったか」を辿るための参照点
- 明らかに不要と確信した時のみ `git push origin --delete <branch>` で削除

---

## 3. コミット分割ルール

実験ブランチ上で、**コード変更とドキュメント更新は必ず別コミットにする**。
これにより、不採用時に「ドキュメントのみマージ」がコストゼロで実現できる。

### 推奨フロー

```
git switch -c exp-XXX

# 1. コード変更
編集 src/modeling/train.py 等
git add src/modeling/train.py
git commit -m "exp: EXP-XXX の介入実装"

# 2. 学習実行
uv run python -m src.modeling.train

# 3. ドキュメント更新
編集 docs/experiment_log.md docs/experiment_proposals.md
git add docs/
git commit -m "docs: EXP-XXX の結果記録と PROP-XXX の状態更新"

# 4. プッシュ
git push -u origin exp-XXX
```

### PR タイトルのプレフィックス

| プレフィックス | 用途 | 例 |
|---|---|---|
| `feat:` / `fix:` | 採用された機能改善 | `feat: 駅単位の地図ページ(P2)を追加` |
| `exp:` | 実験（採用/不採用を問わず実験記録） | `exp: EXP-007 クリップ閾値の緩和(5億)→不採用判定` |
| `docs:` | ドキュメントのみの変更 | `docs: EXP-007 の実験記録` |

`exp:` プレフィックスを使うことで `git log --oneline | grep "exp:"` で実験コミットだけを抽出できる。

---

## 4. 採否判断と PR マージ

### 採否の根拠
PR の採否判断は、[experiment_proposals.md](experiment_proposals.md) の
該当 PROP-XXX で定めた **採用条件** を満たすか否かで行う。
PR 本文にチェックリスト形式で記載し、レビューで明示する。

```markdown
## 採否
- [ ] PROP-XXX の採用条件を満たす → main にマージ
- [x] 採用条件を満たさない → PR をドキュメントのみに絞ってマージ、または Close
```

### 採用時の手順
1. PR をそのままマージ（コード + ドキュメント両方）
2. `experiment_proposals.md` の PROP-XXX の状態を `Done` に更新
3. `experiment_log.md` の Best? 列を新ベストに移す（旧ベストの印は外す）
4. `models/best_params.yaml` の上書きの有無を実験エントリに明記

### 不採用時の手順（重要）
コード変更を `main` に残さないため、以下のいずれかを選ぶ。

#### 方法B: 既存ブランチをドキュメントのみに絞り直す（推奨）
コード変更をベースライン状態に巻き戻すコミットを追加し、PR の差分をドキュメントだけにする。

```bash
git switch exp-XXX

# コード・生成物を main の状態に戻す（実体差分が docs だけになる）
git checkout origin/main -- src/modeling/train.py
git checkout origin/main -- src/preprocessing/run_pipeline.py
git checkout origin/main -- models/best_params.yaml outputs/test_predictions.csv

git add -u
git commit -m "revert: EXP-XXX のコードと生成物を撤回（記録ドキュメントのみマージ用）"

git push origin exp-XXX
```

PR の diff が `docs/model_experiment_*.md` だけになるので、そのままマージする。

> 注意: `main` に既に過去の不採用実験コードが残っている場合、`origin/main` ではなく
> 真のベースラインコミット（例: `6efb65d` のような実験開始前）から `git checkout` する必要がある。

#### 方法C: 新ブランチで docs だけ抜き出す
履歴をより綺麗に保ちたい場合、PR を作り直す。

```bash
git switch -c docs-exp-XXX main

# 実験ブランチから docs ファイルだけ取り込む
git checkout exp-XXX -- docs/experiment_log.md docs/experiment_proposals.md

git add docs/
git commit -m "docs: EXP-XXX の実験記録を追加"
git push origin docs-exp-XXX

# 新 PR を作成
gh pr create --base main --head docs-exp-XXX --title "docs: EXP-XXX の実験記録"

# 元の PR は Close without merge
gh pr close <元のPR番号>
```

#### 方法D: コードもドキュメントも `main` に入れない
ブランチごと残し、PR は Close without merge にする。  
記録は実験ブランチ上にだけ残るため、`main` から実験管理表へは EXP-XXX エントリが
反映されない。**運用上は避ける**（実験管理表が `main` から見て不完全になる）。

### 採否のまとめ

| 採否 | コード | 記録ドキュメント | 生成物 |
|---|---|---|---|
| 採用 | `main` にマージ | `main` にマージ | `.gitignore` |
| 不採用（方法B推奨） | `main` にマージしない | `main` にマージ | `.gitignore` |

---

## 5. 生成物の扱い

実験のたびに `outputs/test_predictions.csv`（約 7,900 行）と
`models/best_params.yaml` / `models/lgbm_model.pkl` が再生成され、
PR の diff を圧迫する（実例: EXP-006 PR で +8,056 / −7,961 行）。

### 推奨 `.gitignore` 追加候補
```
# 実験ごとの再生成物（リポジトリでは管理しない）
outputs/test_predictions.csv
outputs/test_predictions_copy.csv
outputs/test_predictions_*.geojson
models/best_params.yaml
models/lgbm_model.pkl
```

### ベースライン版の固定保管
特定の実験結果（例: 採用版のベースライン予測結果）を残したい場合、
別名でディレクトリを切って退避する。

```
outputs/baseline/
  test_predictions.csv
  best_params.yaml
```

BI ダッシュボードや地図ページが `outputs/test_predictions.csv` を読むため、
ローカルでは常に再生成される運用とする。

### 既にコミット済みの生成物の整理
過去にコミットされた生成物は `.gitignore` を追加するだけでは消えない。
以下で履歴は残しつつ追跡から外す。

```bash
git rm --cached outputs/test_predictions.csv
git rm --cached models/best_params.yaml
git commit -m "chore: 生成物を gitignore に切替"
```

> 履歴サイズを完全に削減したい場合は `git filter-repo` 等が必要だが、
> 個人開発規模では推奨しない（破壊的なため）。

---

## 6. PR 本文テンプレート

### 採用想定の実験 PR
```markdown
## 概要
<EXP-XXX の介入内容と狙いを1〜2文で>

## 変更内容
- コミット <SHA>: <コード変更>
- コミット <SHA>: <ドキュメント更新>

## test スコアと比較
| 指標 | EXP-XXX | EXP-001 比 | (他比較対象) 比 |
|---|---:|---:|---:|
| ... | ... | ... | ... |

## 採否
- [x] PROP-XXX の採用条件を満たす → main にマージ

## レビューポイント
- <注意点・確認してほしい箇所>
```

### 不採用想定の実験 PR
```markdown
## 概要
<EXP-XXX の介入内容と狙いを1〜2文で。不採用判定の旨も明記>

## 変更内容
- コミット <SHA>: <コード変更>
- コミット <SHA>: <ドキュメント更新>

## test スコアと比較
| 指標 | EXP-XXX | EXP-001 比 |
|---|---:|---:|
| ... | ... | ... |

## 採否
- [x] PROP-XXX の採用条件を満たさない → ドキュメントのみマージ

## マージ手順
- 方法B（既存ブランチをドキュメントのみに絞り直す）でコード変更を巻き戻し、ドキュメントだけマージ予定

## 次のアクション
- <次の試行候補や、PROP-XXX の状態更新>
```

---

## 7. 履歴とトレーサビリティ

ルール運用後、以下のように情報が分散・整理される。

| 知りたいこと | 参照先 |
|---|---|
| 現行ベストモデルの構成 | `main` の `src/modeling/train.py` ＋ `models/best_params.yaml`（baseline退避先） |
| 過去にどんな実験を試したか | `main` の [experiment_log.md](experiment_log.md) |
| 未実施の改善アイデア | `main` の [experiment_proposals.md](experiment_proposals.md) |
| 不採用実験の具体的なコード | `exp-XXX` ブランチ（リモートに残す） |
| 採用された変更の経緯 | `main` の git log（`feat:` / `fix:` プレフィックスで絞り込み） |
| 実験そのものの経緯 | `main` の git log（`exp:` / `docs:` プレフィックスで絞り込み） |

```bash
# 採用された改善だけを見る
git log --oneline --grep "^feat:\|^fix:"

# 実験の足跡を見る
git log --oneline --grep "^exp:\|^docs: EXP"
```

---

## 8. 既存の負債への対応（任意）

本ルールを導入する前にマージされた不採用実験（EXP-005 など）が `main` に残る場合、
以下のいずれかで対処する。

| 対応 | 内容 | コスト |
|---|---|---|
| A. 放置 | 後続採用実験でコードが上書きされるのを待つ | 小（ノイズが一定期間残る） |
| B. 撤回 PR | 不採用実験のコード変更を revert する PR を1本立てる | 中 |
| C. リセット | 真のベースラインまで `main` を巻き戻す | 大（破壊的） |

通常は A で十分。B は「不採用実験コードが新規実装の妨げになっている」場合のみ。

---

## 9. チェックリスト（運用開始時）

新しい実験を始める前に、以下を確認する。

- [ ] `.gitignore` に生成物のパターンを追加済み
- [ ] 既にコミット済みの生成物は `git rm --cached` で追跡解除済み
- [ ] 実験案管理表（[experiment_proposals.md](experiment_proposals.md)）に該当案がある
- [ ] 採用条件が明文化されている
- [ ] ブランチ命名と PR タイトルのプレフィックス規約を把握している
- [ ] コードとドキュメントを別コミットにする運用を理解している

---

最終更新: 2026-06-02
