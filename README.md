# cnt-cvf-simulation

真空ろ過法（CVF: Centrifugal Vacuum Filtration）によるCNT配向プロセスのシミュレーション。

実験条件（drop間隔・膜構造・CNT長さ分布）と配向度（ラマンGピーク比）の関係を流体力学的にモデル化する。

## クイックスタート

```bash
pip install -r requirements.txt
```

開発仕様は `CLAUDE.md` を参照。

## ディレクトリ構成

```
cnt-cvf-simulation/
├── CLAUDE.md           # 開発仕様（必読）
├── README.md
├── requirements.txt
├── src/cnt_cvf/        # コアロジック
├── scripts/            # 実行スクリプト
├── notebooks/          # 探索的分析
├── tests/              # ユニットテスト
├── data/               # 実験データ
└── results/            # シミュ結果（git管理外）
```

## 開発の進め方

`CLAUDE.md` の「開発の進め方」セクションを参照。Step 1 → Step 5 の段階的実装で、各ステップの終わりに出力を確認してから次に進む。
