# archive/ — Phase 1 の旧モデル（retired）

このディレクトリは **Phase 1 の流体配向モデル** のコードを保管している。
**Phase 2 では使用しない**。新規開発時に参照・再利用してはならない。

## なぜ archive されたか

Phase 1 は CVF の配向機構を「流体配向（Jeffery + Brownian）」として
モデル化していた。具体的には：

- 点シンク流れ場で局所せん断率 γ̇(r,z) を計算
- 各 CNT に Pe = γ̇/D_r を割り当て、Langevin で定常配向を計算
- 多層モデル + 長さ分布の MC で G ピーク比を予測

しかし結果として：
- 達成可能範囲（μ_L ≤ 500 nm）で予測上限 G ≈ 1.4
- 実測値 1.78（8min）、1.59（15min）、3.4（短尺スタート）を**原理的に
  説明できない**ことが判明
- 文献調査の結果、CVF の配向機構は流体配向ではなく **2D 液晶相転移**
  であることが確立されていた（Komatsu 2020, He 2016, Gao & Kono 2019）

## Phase 1 で得た有用な知見（Phase 2 でも有効）

1. **単純な流体配向では実測を説明できない** — これは Phase 2 の
   出発点となる重要な事実。論文では「流体配向だけでは不十分である
   ことを定量的に示した」として書ける。
2. **長さ分布だけでは観測される配向度の差を説明できない** — 8min と
   15min/30min のデータを比較した結果。長さ以外の因子（面密度・単離度
   など）が支配的であることの証拠。
3. **長さ分布データの整理** — 3 サンプル（8/15/30 min）の AFM 計測と
   配向度との対応関係を確立した。Phase 2 でもこのデータをそのまま使う。

## 含まれるファイル

- `flow_field.py` — 点シンク流れ場、γ̇(r,z) の計算
- `dynamics.py` — Langevin 方程式、回転拡散
- `runner.py` — モンテカルロ実行
- `scripts/` — 旧 Step 2/3 のスクリプト群

## Phase 2 で残したコード（archive 対象外）

以下は `src/cnt_cvf/` に残してある（Phase 2 でも有効）：

- `config.py` — 物理定数・固定パラメータ
- `orientation.py` — Kawai 式（角度分布 → G 比の変換器、依然として正しい）
- `length_dist.py` — 長さ分布のサンプリング

`tests/test_orientation.py` と `tests/test_length_dist.py` も残す。
それ以外の Phase 1 関連テスト（`test_dynamics.py`, `test_flow_field.py`,
`test_runner.py`）は `archive/tests/` に移動。

## results/ について

`results/step1_*` から `results/step3c_*` までの旧 Phase 1 の出力は
そのまま `results/` 以下に残してある（削除しない）。論文執筆時に
「流体配向モデルでの予測」として比較対象になる可能性があるため。
Phase 2 の新規出力は別ディレクトリ（`results/phase2_step1_*` など）
に作成すること。
