<h1 align="center">Strata-2080TI</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · **Español** · [Português](README.pt-BR.md)

## Este fork: 8x RTX 2080 Ti 22 GB

Strata-2080TI está basado en [Niko1221/Strata](https://github.com/Niko1221/Strata) y optimizado para un servidor con
**8x NVIDIA RTX 2080 Ti 22 GB** (Turing, sm_75). Para instalar y usar Strata, consulta el README de [Niko1221/Strata](https://github.com/Niko1221/Strata).

**Equipo de prueba:** 8x RTX 2080 Ti 22 GB solo por PCIe (sin NVLink, sin P2P), las tarjetas repartidas en dos nodos
NUMA, 377 GB de RAM, CUDA 12.8. Modelo: [Unsloth Q8_0](docs/UNSLOTH_Q8.md), dividido por capas entre las 8 tarjetas,
todos los expertos en VRAM (99,8 % de aciertos de caché). Motor 0.1.40.3, medido el 2026-10-11.

**Qué cambió:**

- `--split-chunks N` (nueva opción del motor): en una división por capas, el prompt se lee en unos N trozos de
  2048-4096 tokens, así los búferes ya no toman prestada la mayor parte de la caché de expertos de cada tarjeta. Antes,
  `--prefill auto:32768` tomaba 2.295 de las 3.072 ranuras de cada tarjeta y los expertos volvían a cruzar PCIe en
  cada prompt.
- Los hilos de prompt de cada etapa ya no heredan el núcleo fijado del hilo principal. 35 hilos del motor compartían
  la CPU 0 y las GPU la esperaban un 14 % del tiempo de GPU de un prompt (8,4 s -> 0,4 s).
- Ajustes para Turing: `STRATA_BF16_TC=1` (núcleos tensor BF16), `STRATA_PREFILL_ZC=1` (expertos residentes leídos en
  su sitio, opcional, no idéntico bit a bit), `"parallel": 4` con `--batch-groups 4`.

**Lee tu prompt** (tokens/s):

| Prompt | Antes, `--prefill auto:32768` | Este fork | Mejora vs. `auto:32768` |
| ---: | ---: | ---: | ---: |
| 4K | 713-751 | **1.080** | 1,5x |
| 16K | 891-908 | **2.240** | 2,5x |
| 32K | 809-883 | **2.920** | 3,4x |
| 100K | — | **3.974** | — |

**Escribe respuestas:** 57-60 tokens/s con una petición (sin cambios: la decodificación está limitada por la GPU, las
8 tarjetas calculan sus capas una tras otra). **8 peticiones a la vez:** 136 tokens/s en total, frente a 50 tokens/s
una por una.

La columna «Antes» es el proyecto original con `--prefill auto:32768`, dos peticiones cada una, de [docs/MULTI_GPU.md](docs/MULTI_GPU.md#what-each-card-holds) (allí
están también las cifras paso a paso). Probado en estas tarjetas sin ganancia: `--spec 6` (más lento),
`--mtp-q4 all`, `STRATA_HC_Q8`.

Los ajustes, añadidos a la configuración que escribió setup (`strata-<model>.json`):

```json
"args": ["...", "--prefill", "auto", "--split-chunks", "4", "--batch-groups", "4"],
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_ZC": "1"},
"parallel": 4
```
