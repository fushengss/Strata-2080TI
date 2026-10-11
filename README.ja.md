<h1 align="center">Strata-2080TI</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語** · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

## このフォーク：8× RTX 2080 Ti 22 GB

Strata-2080TI は [Niko1221/Strata](https://github.com/Niko1221/Strata) をベースに、
**NVIDIA RTX 2080 Ti 22 GB を 8 枚**（Turing、sm_75）積んだ 1 台のサーバー向けに最適化したものです。Strata のインストールと使い方は [Niko1221/Strata](https://github.com/Niko1221/Strata) の README を参照してください。

**テスト機：** 8× RTX 2080 Ti 22 GB、PCIe のみ（NVLink なし、P2P なし）、カードは 2 つの NUMA ノードに分かれ、メモリ 377 GB、
CUDA 12.8。モデル：[Unsloth Q8_0](docs/UNSLOTH_Q8.md)、8 枚すべてにレイヤー単位で分割、全エキスパートが VRAM 上（キャッシュヒット率 99.8%）。
エンジン 0.1.40.3、2026-10-11 測定。

**変更点：**

- `--split-chunks N`（新しいエンジンオプション）：レイヤー分割時、プロンプトを 2048-4096 トークンの約 N 個のチャンクで読みます。
  チャンク用バッファが各カードのエキスパートキャッシュの大半を借りなくなります。以前は `--prefill auto:32768` が各カード 3,072 スロットのうち
  2,295 を借り、プロンプトを読むたびにエキスパートが再び PCIe を通っていました。
- 各ステージのプロンプト用スレッドが、ホストスレッドの固定コアを引き継がなくなりました。35 のエンジンスレッドが CPU 0 を共有し、
  GPU はプロンプトの GPU 時間の 14% をその待ちに使っていました（8.4 秒 -> 0.4 秒）。
- Turing 向けの設定：`STRATA_BF16_TC=1`（BF16 テンソルコア）、`STRATA_PREFILL_ZC=1`（常駐エキスパートをその場で読む。オプトイン、
  ビット単位では一致しません）、`"parallel": 4` と `--batch-groups 4`。

**プロンプトを読む速さ**（tokens/s）：

| プロンプト | 変更前、`--prefill auto:32768` | このフォーク | `auto:32768` 比 |
| ---: | ---: | ---: | ---: |
| 4K | 713-751 | **1,080** | 1.5 倍 |
| 16K | 891-908 | **2,240** | 2.5 倍 |
| 32K | 809-883 | **2,920** | 3.4 倍 |
| 100K | — | **3,974** | — |

**回答を書く速さ：** 1 リクエストで 57-60 tokens/s（変化なし：デコードは GPU 律速で、8 枚が順番に自分のレイヤーを処理します）。
**8 リクエスト同時：** 合計 136 tokens/s、1 件ずつ処理すると 50 tokens/s。

「変更前」の列は上流版 `--prefill auto:32768` で各 2 リクエスト、[docs/MULTI_GPU.md](docs/MULTI_GPU.md#what-each-card-holds) より（変更ごとの数値もそこにあります）。
このカードで試して効果がなかったもの：`--spec 6`（遅くなる）、`--mtp-q4 all`、`STRATA_HC_Q8`。

設定は、setup が書いた設定ファイル（`strata-<model>.json`）に追加します：

```json
"args": ["...", "--prefill", "auto", "--split-chunks", "4", "--batch-groups", "4"],
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_ZC": "1"},
"parallel": 4
```
