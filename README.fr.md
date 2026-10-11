<h1 align="center">Strata-2080TI</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · **Français** · [Español](README.es.md) · [Português](README.pt-BR.md)

## Ce fork : 8x RTX 2080 Ti 22 Go

Strata-2080TI est basé sur [Niko1221/Strata](https://github.com/Niko1221/Strata) et optimisé pour un serveur équipé de
**8x NVIDIA RTX 2080 Ti 22 Go** (Turing, sm_75). Installer et utiliser Strata : voir le README de [Niko1221/Strata](https://github.com/Niko1221/Strata).

**Machine de test :** 8x RTX 2080 Ti 22 Go en PCIe seulement (pas de NVLink, pas de P2P), cartes réparties sur deux
nœuds NUMA, 377 Go de RAM, CUDA 12.8. Modèle : [Unsloth Q8_0](docs/UNSLOTH_Q8.md), réparti par couches sur les 8
cartes, tous les experts en VRAM (99,8 % de succès du cache). Moteur 0.1.40.3, mesuré le 2026-10-11.

**Ce qui a changé :**

- `--split-chunks N` (nouvelle option du moteur) : sur une répartition par couches, un prompt est lu en environ N
  morceaux de 2048-4096 tokens, et les tampons n'empruntent plus la plupart du cache d'experts de chaque carte. Avant,
  `--prefill auto:32768` empruntait 2 295 des 3 072 emplacements de chaque carte, et les experts repassaient par PCIe à
  chaque prompt.
- Les threads de prompt des étages n'héritent plus du cœur épinglé du thread hôte. 35 threads du moteur partageaient
  le CPU 0 et les GPU l'attendaient pendant 14 % du temps GPU d'un prompt (8,4 s -> 0,4 s).
- Réglages pour Turing : `STRATA_BF16_TC=1` (cœurs tenseurs BF16), `STRATA_PREFILL_ZC=1` (experts résidents lus sur
  place, optionnel, pas identique au bit près), `"parallel": 4` avec `--batch-groups 4`.

**Lit votre prompt** (tokens/s) :

| Prompt | Avant, `--prefill auto:32768` | Ce fork | Gain vs. `auto:32768` |
| ---: | ---: | ---: | ---: |
| 4K | 713-751 | **1 080** | 1,5x |
| 16K | 891-908 | **2 240** | 2,5x |
| 32K | 809-883 | **2 920** | 3,4x |
| 100K | — | **3 974** | — |

**Écrit les réponses :** 57-60 tokens/s pour une requête (inchangé : le décodage est limité par le GPU, les 8 cartes
calculent leurs couches l'une après l'autre). **8 requêtes à la fois :** 136 tokens/s au total, contre 50 tokens/s
une par une.

La colonne « Avant » est le projet d'origine avec `--prefill auto:32768`, deux requêtes chacune, tirées de
[docs/MULTI_GPU.md](docs/MULTI_GPU.md#what-each-card-holds) (les chiffres étape par étape y sont aussi). Essayé sur ces
cartes sans gain : `--spec 6` (plus lent), `--mtp-q4 all`, `STRATA_HC_Q8`.

Les réglages, à ajouter à la configuration écrite par setup (`strata-<model>.json`) :

```json
"args": ["...", "--prefill", "auto", "--split-chunks", "4", "--batch-groups", "4"],
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_ZC": "1"},
"parallel": 4
```
