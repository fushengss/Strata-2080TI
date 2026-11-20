#!/usr/bin/env python3
"""Apply the small diagnostic-only edits to upstream native_expert_bench.cpp.

Run once on a fresh copy; writes only the supplied temporary source path.
The two kernel switches are process-static: compare raw dumps between processes.
"""
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()

def replace(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit(f'expected exactly one occurrence: {old[:70]}')
    s = s.replace(old, new)

replace('const int iters = argc > 7 ? std::atoi(argv[7]) : 200;',
        'const int iters = std::max(1, argc > 7 ? std::atoi(argv[7]) : 50);\n'
        '    const int rounds = std::max(1, std::getenv("BENCH_ROUNDS") ? std::atoi(std::getenv("BENCH_ROUNDS")) : 3);')
replace('const int NTOK = 8;', 'const int NTOK = 4; // diagnostic: production spec=3 verifies at most four tokens')
replace('std::mt19937 rng(17 + l);',
        'std::mt19937 rng((std::getenv("BENCH_SEED") ? std::atoi(std::getenv("BENCH_SEED")) : 17) + l);')
replace('        size_t ndiff = 0;', '''        size_t nonfinite = 0;
        for (float v : got) nonfinite += !std::isfinite(v);
        for (float v : ref) nonfinite += !std::isfinite(v);
        std::printf("finite_check layer=%d nonfinite=%zu values=%zu\\n", l, nonfinite, got.size() + ref.size());
        if (nonfinite) return 3;
        if (const char* prefix = std::getenv("BENCH_DUMP_PREFIX")) {
            const std::string path = std::string(prefix) + ".l" + std::to_string(l) + ".g" + std::to_string(G) + ".t" + std::to_string(T) + ".bin";
            FILE* f = std::fopen(path.c_str(), "wb");
            if (!f) { std::perror(path.c_str()); return 4; }
            const size_t wrote = std::fwrite(got.data(), sizeof(float), got.size(), f);
            const int closed = std::fclose(f);
            if (wrote != got.size() || closed) { std::fprintf(stderr, "dump failed: %s\\n", path.c_str()); return 4; }
            std::printf("dump %s %zu bytes\\n", path.c_str(), got.size() * sizeof(float));
        }
        size_t ndiff = 0;''')
replace('for (int round = 0; round < 3; ++round)', 'for (int round = 0; round < rounds; ++round)')
replace('for (int p = 0; p < 3; ++p) us[k][p]', 'for (int p = 0; p < 1; ++p) us[k][p]')
old = '''        std::printf("layer %2d %-7s/%-6s G %d T %d | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s)"
                    " | mode %d: all %7.1f us  gu %7.1f us (%3.0f GB/s)  down %6.1f us (%3.0f GB/s) | %s\\n",
                    l, ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), G, T,
                    mref, us[0][0], us[0][1], gu_b / us[0][1] / 1e3, us[0][2], d_b / us[0][2] / 1e3,
                    mode, us[1][0], us[1][1], gu_b / us[1][1] / 1e3, us[1][2], d_b / us[1][2] / 1e3,
                    ndiff ? "DIFFERS" : "bitwise equal");'''
new = '''        std::printf("layer %2d %-7s/%-6s G %d T %d | mode %d: all %.3f us | mode %d: all %.3f us | in-process modes %s | unique-weight GB/s %.3f\\n",
                    l, ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), G, T,
                    mref, us[0][0], mode, us[1][0], ndiff ? "DIFFERS" : "equal (NOT env-switch parity)",
                    (gu_b + d_b) / us[1][0] / 1e3);'''
replace(old, new)
p.write_text(s)
