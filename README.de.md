<h1 align="center">Strata-2080TI</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **Deutsch** · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

## Dieser Fork: 8x RTX 2080 Ti 22 GB

Strata-2080TI basiert auf [Niko1221/Strata](https://github.com/Niko1221/Strata) und ist auf einen Server mit
**8x NVIDIA RTX 2080 Ti 22 GB** (Turing, sm_75) abgestimmt. Installation und Bedienung von Strata: siehe die README von [Niko1221/Strata](https://github.com/Niko1221/Strata).

**Testrechner:** 8x RTX 2080 Ti 22 GB nur über PCIe (kein NVLink, kein P2P), die Karten auf zwei NUMA-Knoten verteilt,
377 GB RAM, CUDA 12.8. Modell: [Unsloth Q8_0](docs/UNSLOTH_Q8.md), schichtweise auf alle 8 Karten verteilt, jeder
Experte im VRAM (99,8 % Cache-Treffer). Engine 0.1.40.3, gemessen am 2026-10-11.

**Was sich geändert hat:**

- `--split-chunks N` (neue Engine-Option): Bei einer Schichtaufteilung wird ein Prompt in etwa N Stücken zu 2048-4096
  Tokens gelesen, sodass die Puffer nicht mehr den Großteil des Experten-Caches jeder Karte leihen. Vorher lieh
  `--prefill auto:32768` 2.295 der 3.072 Plätze jeder Karte, und die Experten liefen bei jedem Prompt erneut über PCIe.
- Die Prompt-Threads der Stufen erben nicht mehr den festgelegten Kern des Host-Threads. 35 Engine-Threads teilten
  sich CPU 0, und die GPUs warteten 14 % der GPU-Zeit eines Prompts darauf (8,4 s -> 0,4 s).
- Einstellungen für Turing: `STRATA_BF16_TC=1` (BF16-Tensor-Cores), `STRATA_PREFILL_ZC=1` (residente Experten an Ort
  und Stelle gelesen, optional, nicht bitgleich), `"parallel": 4` mit `--batch-groups 4`.

**Liest deinen Prompt** (Tokens/s):

| Prompt | Vorher, `--prefill auto:32768` | Dieser Fork | Faktor ggü. `auto:32768` |
| ---: | ---: | ---: | ---: |
| 4K | 713-751 | **1.080** | 1,5x |
| 16K | 891-908 | **2.240** | 2,5x |
| 32K | 809-883 | **2.920** | 3,4x |
| 100K | — | **3.974** | — |

**Schreibt Antworten:** 57-60 Tokens/s für eine Anfrage (unverändert: das Dekodieren ist GPU-gebunden, die 8 Karten
rechnen ihre Schichten nacheinander). **8 Anfragen gleichzeitig:** 136 Tokens/s insgesamt, gegenüber 50 Tokens/s
nacheinander.

Die „Vorher“-Spalte ist das Originalprojekt mit `--prefill auto:32768`, je zwei Anfragen, aus [docs/MULTI_GPU.md](docs/MULTI_GPU.md#what-each-card-holds) (dort auch
die Zahlen Schritt für Schritt). Auf diesen Karten ohne Gewinn getestet: `--spec 6` (langsamer), `--mtp-q4 all`,
`STRATA_HC_Q8`.

Die Einstellungen, ergänzt in der Konfiguration, die Setup geschrieben hat (`strata-<model>.json`):

```json
"args": ["...", "--prefill", "auto", "--split-chunks", "4", "--batch-groups", "4"],
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_ZC": "1"},
"parallel": 4
```
