# 100k Q5 A/B comparison

API usage gives exact token totals. Client decode is a first-to-last SSE arrival estimate; engine decode is only shown for a unique matching prompt/output-token record. Cold timings require confirmed zero reuse.

| Arm | Request | Prompt / output | Cold zero reuse | Engine read tokens / ms | Prefill tok/s | TTFT s | Client decode est. tok/s | Engine full request decode tok/s | Finish |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| balanced | cold-1 | 101137 / 768 | yes | 101137 / 15495 | 6526.90 | 15.98 | 134.35 | 127.40 | length |
| balanced | cold-2 | 101137 / 768 | yes | 101137 / 17498 | 5780.00 | 17.69 | 136.94 | 136.60 | length |
| balanced | cold-3 | 101111 / 512 | yes | 101111 / 17712 | 5708.50 | 17.90 | 81.63 | 81.50 | length |
| balanced | append-600 | 102543 / 256 | append | 638 / 4141 | 154.10 | 4.25 | 104.70 | 104.30 | length |
| baseline | cold-1 | 101137 / 768 | yes | 101137 / 14486 | 6981.90 | 14.93 | 133.36 | 127.60 | length |
| baseline | cold-2 | 101137 / 768 | yes | 101137 / 16837 | 6006.80 | 17.02 | 137.58 | 137.30 | length |
| baseline | cold-3 | 101111 / 512 | yes | 101111 / 16777 | 6026.80 | 16.95 | 81.29 | 81.20 | length |
| baseline | append-600 | 102543 / 256 | append | 638 / 4199 | 152.00 | 4.31 | 105.53 | 105.10 | length |
| combo | cold-1 | 101137 / 768 | yes | 101137 / 14468 | 6990.30 | 14.88 | 134.72 | 128.90 | length |
| combo | cold-2 | 101137 / 768 | yes | 101137 / 16726 | 6046.60 | 16.92 | 138.54 | 138.10 | length |
| combo | cold-3 | 101111 / 512 | yes | 101111 / 16735 | 6041.90 | 16.92 | 81.46 | 81.40 | length |
| combo | append-600 | 102543 / 256 | append | 638 / 4156 | 153.50 | 4.26 | 106.02 | 105.60 | length |
| groups8 | cold-1 | 101137 / 768 | yes | 101137 / 14506 | 6972.10 | 14.98 | 133.29 | 126.60 | length |
| groups8 | cold-2 | 101137 / 768 | yes | 101137 / 16772 | 6030.20 | 16.96 | 137.55 | 137.20 | length |
| groups8 | cold-3 | 101111 / 512 | yes | 101111 / 16773 | 6028.20 | 16.96 | 81.26 | 81.20 | length |
| groups8 | append-600 | 102543 / 256 | append | 638 / 4210 | 151.50 | 4.31 | 105.53 | 105.20 | length |
| v2k | cold-1 | 101137 / 768 | yes | 101137 / 14493 | 6978.30 | 14.97 | 134.91 | 128.30 | length |
| v2k | cold-2 | 101137 / 768 | yes | 101137 / 16735 | 6043.40 | 16.90 | 138.88 | 138.40 | length |
| v2k | cold-3 | 101111 / 512 | yes | 101111 / 16718 | 6048.10 | 16.88 | 81.51 | 81.40 | length |
| v2k | append-600 | 102543 / 256 | append | 638 / 4140 | 154.10 | 4.24 | 106.29 | 105.80 | length |

| Arm | Cohort | Wall s | Output tokens | Aggregate tok/s over wall | Per-request decode est. range | Common decode s | Common chunks/s (not tok/s) | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | c6-long | 55.12 | 6144 | 111.47 | 21.84–40.89 | 24.64 | 219.13 | 0 |
| baseline | c8-short | 29.68 | 8192 | 276.00 | 37.51–43.81 | 23.22 | 306.97 | 0 |
| combo | c6-long | 52.45 | 6144 | 117.14 | 23.09–47.54 | 21.24 | 254.15 | 0 |
| combo | c8-short | 33.99 | 8192 | 241.01 | 32.56–37.06 | 27.31 | 260.56 | 0 |
| groups8 | c6-long | 51.52 | 6144 | 119.24 | 23.44–48.60 | 20.76 | 260.09 | 0 |
| groups8 | c8-short | 33.17 | 8192 | 246.97 | 33.33–38.09 | 26.65 | 266.89 | 0 |
| v2k | c6-long | 55.39 | 6144 | 110.92 | 21.67–39.77 | 25.39 | 212.65 | 0 |
| v2k | c8-short | 30.55 | 8192 | 268.15 | 36.43–42.27 | 24.07 | 296.23 | 0 |

Text parity reference: `baseline`. Fixed single requests require exact text equality when request hashes agree. Concurrent differences are reported without declaring a correctness failure. Text checks cannot establish finite internal logits.

| Candidate | Request | Same messages | Same full request | Same output hash | Result | First difference (offset / line / col) |
|---|---|---|---|---|---|---|
| balanced | append-600 | True | True | True | exact_text_match | — |
| balanced | c6-long-1 | — | — | — | missing_request | — |
| balanced | c6-long-2 | — | — | — | missing_request | — |
| balanced | c6-long-3 | — | — | — | missing_request | — |
| balanced | c6-long-4 | — | — | — | missing_request | — |
| balanced | c6-long-5 | — | — | — | missing_request | — |
| balanced | c6-long-6 | — | — | — | missing_request | — |
| balanced | c8-short-1 | — | — | — | missing_request | — |
| balanced | c8-short-2 | — | — | — | missing_request | — |
| balanced | c8-short-3 | — | — | — | missing_request | — |
| balanced | c8-short-4 | — | — | — | missing_request | — |
| balanced | c8-short-5 | — | — | — | missing_request | — |
| balanced | c8-short-6 | — | — | — | missing_request | — |
| balanced | c8-short-7 | — | — | — | missing_request | — |
| balanced | c8-short-8 | — | — | — | missing_request | — |
| balanced | cold-1 | True | True | True | exact_text_match | — |
| balanced | cold-2 | True | True | True | exact_text_match | — |
| balanced | cold-3 | True | True | True | exact_text_match | — |
| combo | append-600 | True | True | True | exact_text_match | — |
| combo | c6-long-1 | True | True | True | exact_text_match | — |
| combo | c6-long-2 | True | True | True | exact_text_match | — |
| combo | c6-long-3 | True | True | True | exact_text_match | — |
| combo | c6-long-4 | True | True | True | exact_text_match | — |
| combo | c6-long-5 | True | True | True | exact_text_match | — |
| combo | c6-long-6 | True | True | True | exact_text_match | — |
| combo | c8-short-1 | True | True | True | exact_text_match | — |
| combo | c8-short-2 | True | True | True | exact_text_match | — |
| combo | c8-short-3 | True | True | True | exact_text_match | — |
| combo | c8-short-4 | True | True | True | exact_text_match | — |
| combo | c8-short-5 | True | True | True | exact_text_match | — |
| combo | c8-short-6 | True | True | True | exact_text_match | — |
| combo | c8-short-7 | True | True | True | exact_text_match | — |
| combo | c8-short-8 | True | True | True | exact_text_match | — |
| combo | cold-1 | True | True | True | exact_text_match | — |
| combo | cold-2 | True | True | True | exact_text_match | — |
| combo | cold-3 | True | True | True | exact_text_match | — |
| groups8 | append-600 | True | True | True | exact_text_match | — |
| groups8 | c6-long-1 | True | True | True | exact_text_match | — |
| groups8 | c6-long-2 | True | True | True | exact_text_match | — |
| groups8 | c6-long-3 | True | True | True | exact_text_match | — |
| groups8 | c6-long-4 | True | True | True | exact_text_match | — |
| groups8 | c6-long-5 | True | True | True | exact_text_match | — |
| groups8 | c6-long-6 | True | True | True | exact_text_match | — |
| groups8 | c8-short-1 | True | True | True | exact_text_match | — |
| groups8 | c8-short-2 | True | True | True | exact_text_match | — |
| groups8 | c8-short-3 | True | True | True | exact_text_match | — |
| groups8 | c8-short-4 | True | True | True | exact_text_match | — |
| groups8 | c8-short-5 | True | True | True | exact_text_match | — |
| groups8 | c8-short-6 | True | True | True | exact_text_match | — |
| groups8 | c8-short-7 | True | True | True | exact_text_match | — |
| groups8 | c8-short-8 | True | True | True | exact_text_match | — |
| groups8 | cold-1 | True | True | True | exact_text_match | — |
| groups8 | cold-2 | True | True | True | exact_text_match | — |
| groups8 | cold-3 | True | True | True | exact_text_match | — |
| v2k | append-600 | True | True | True | exact_text_match | — |
| v2k | c6-long-1 | True | True | True | exact_text_match | — |
| v2k | c6-long-2 | True | True | True | exact_text_match | — |
| v2k | c6-long-3 | True | True | True | exact_text_match | — |
| v2k | c6-long-4 | True | True | True | exact_text_match | — |
| v2k | c6-long-5 | True | True | True | exact_text_match | — |
| v2k | c6-long-6 | True | True | True | exact_text_match | — |
| v2k | c8-short-1 | True | True | True | exact_text_match | — |
| v2k | c8-short-2 | True | True | True | exact_text_match | — |
| v2k | c8-short-3 | True | True | True | exact_text_match | — |
| v2k | c8-short-4 | True | True | True | exact_text_match | — |
| v2k | c8-short-5 | True | True | True | exact_text_match | — |
| v2k | c8-short-6 | True | True | True | exact_text_match | — |
| v2k | c8-short-7 | True | True | True | exact_text_match | — |
| v2k | c8-short-8 | True | True | True | exact_text_match | — |
| v2k | cold-1 | True | True | True | exact_text_match | — |
| v2k | cold-2 | True | True | True | exact_text_match | — |
| v2k | cold-3 | True | True | True | exact_text_match | — |

First-difference context and full hashes are in `ab-comparison.json`.

- balanced: complete, 4 requests, 0 request errors, 0 invalid saved-output hashes, 0 CUDA/memory-error evidence lines.
- baseline: complete, 18 requests, 0 request errors, 0 invalid saved-output hashes, 0 CUDA/memory-error evidence lines.
- combo: complete, 18 requests, 0 request errors, 0 invalid saved-output hashes, 0 CUDA/memory-error evidence lines.
- groups8: complete, 18 requests, 0 request errors, 0 invalid saved-output hashes, 0 CUDA/memory-error evidence lines.
- v2k: complete, 18 requests, 0 request errors, 0 invalid saved-output hashes, 0 CUDA/memory-error evidence lines.
