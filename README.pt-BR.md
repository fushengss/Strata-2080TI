<h1 align="center">Strata-2080TI</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · **Português**

## Este fork: 8x RTX 2080 Ti 22 GB

O Strata-2080TI é baseado no [Niko1221/Strata](https://github.com/Niko1221/Strata) e otimizado para um servidor com
**8x NVIDIA RTX 2080 Ti 22 GB** (Turing, sm_75). Para instalar e usar o Strata, veja o README de [Niko1221/Strata](https://github.com/Niko1221/Strata).

**Máquina de teste:** 8x RTX 2080 Ti 22 GB só por PCIe (sem NVLink, sem P2P), placas divididas em dois nós NUMA,
377 GB de RAM, CUDA 12.8. Modelo: [Unsloth Q8_0](docs/UNSLOTH_Q8.md), dividido por camadas entre as 8 placas, todos
os experts na VRAM (99,8% de acertos no cache). Motor 0.1.40.3, medido em 2026-10-11.

**O que mudou:**

- `--split-chunks N` (nova opção do motor): numa divisão por camadas, o prompt é lido em cerca de N pedaços de
  2048-4096 tokens, e os buffers deixam de pegar emprestada a maior parte do cache de experts de cada placa. Antes,
  `--prefill auto:32768` pegava 2.295 dos 3.072 slots de cada placa e os experts passavam de novo pelo PCIe a cada
  prompt.
- As threads de prompt de cada estágio não herdam mais o núcleo fixado da thread principal. 35 threads do motor
  dividiam a CPU 0 e as GPUs esperavam por ela 14% do tempo de GPU de um prompt (8,4 s -> 0,4 s).
- Ajustes para Turing: `STRATA_BF16_TC=1` (tensor cores BF16), `STRATA_PREFILL_ZC=1` (experts residentes lidos no
  lugar, opcional, não idêntico bit a bit), `"parallel": 4` com `--batch-groups 4`.

**Lê seu prompt** (tokens/s):

| Prompt | Antes, `--prefill auto:32768` | Este fork | Ganho vs. `auto:32768` |
| ---: | ---: | ---: | ---: |
| 4K | 713-751 | **1.080** | 1,5x |
| 16K | 891-908 | **2.240** | 2,5x |
| 32K | 809-883 | **2.920** | 3,4x |
| 100K | — | **3.974** | — |

**Escreve respostas:** 57-60 tokens/s com uma requisição (sem mudança: a decodificação é limitada pela GPU, as 8
placas calculam suas camadas uma depois da outra). **8 requisições ao mesmo tempo:** 136 tokens/s no total, contra
50 tokens/s uma de cada vez.

A coluna "Antes" é o projeto original com `--prefill auto:32768`, duas requisições cada, de [docs/MULTI_GPU.md](docs/MULTI_GPU.md#what-each-card-holds) (os números
passo a passo também estão lá). Testado nessas placas sem ganho: `--spec 6` (mais lento), `--mtp-q4 all`,
`STRATA_HC_Q8`.

Os ajustes, adicionados à configuração que o setup escreveu (`strata-<model>.json`):

```json
"args": ["...", "--prefill", "auto", "--split-chunks", "4", "--batch-groups", "4"],
"env": {"STRATA_BF16_TC": "1", "STRATA_PREFILL_ZC": "1"},
"parallel": 4
```
