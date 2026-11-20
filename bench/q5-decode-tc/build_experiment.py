#!/usr/bin/env python3
"""Build the tracing engine and operator replay against the selected SM75 release."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess


RELEASE = Path("/srv/bench-20261118-decode-throughput")
BUILD = RELEASE / "build"
SOURCE = RELEASE / "source"
ENGINE_SHA = "d4cdb02e868a05c96f032d8c29993704bc72581d5af9c71269c7f8f03c58e60e"
VERIFY = "src/core/verify.cpp"
VERIFY_KERNELS = "src/kernels/cuda/verify_kernels.cu"
QSA_KERNELS = "src/kernels/cuda/qsa_prompt_attn.cu"
VERIFY_OBJECT = "CMakeFiles/strata_engine.dir/" + VERIFY + ".o"
CUDA_OBJECT = "CMakeFiles/strata_kernels.dir/" + VERIFY_KERNELS + ".o"
QSA_OBJECT = "CMakeFiles/strata_kernels.dir/" + QSA_KERNELS + ".o"
GENERATE = "src/program/generate.cpp"
GENERATE_OBJECT = "CMakeFiles/strata.dir/" + GENERATE + ".o"
BENCH_OBJECT = "CMakeFiles/native_expert_bench.dir/src/kernels/native_expert_bench.cpp.o"


def template(target):
    result = subprocess.check_output(["ninja", "-t", "commands", target], cwd=BUILD, text=True)
    return shlex.split(result.splitlines()[-1])


def replace_option(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def compile_as(target, source, output, overlay):
    command = template(target)
    original = Path(command[command.index("-c") + 1])
    replace_option(command, "-c", source)
    replace_option(command, "-o", output)
    replace_option(command, "-MT", output)
    replace_option(command, "-MF", output.with_suffix(".d"))
    command[1:1] = [f"-I{overlay / 'include'}", f"-I{original.parent}"]
    subprocess.run(command, cwd=BUILD, check=True)
    return output


def link_as(target, output, replacements, extra_object=None):
    command = template(target)
    assert command[:2] == [":", "&&"] and command[-2:] == ["&&", ":"]
    command = [str(replacements.get(word, word)) for word in command[2:-2]]
    replace_option(command, "-o", output)
    if extra_object is not None:
        command.insert(command.index("libstrata_kernels.a"), str(extra_object))
    subprocess.run(command, cwd=BUILD, check=True)


def replace_archive(directory, member, name):
    copy = directory / name
    shutil.copy2(BUILD / name, copy)
    subprocess.run(["ar", "r", str(copy), str(member)], check=True)
    return copy


def capture(overlay, output, include_qsa=False):
    verify = overlay / VERIFY
    cuda = overlay / VERIFY_KERNELS
    with ThreadPoolExecutor(max_workers=4) as pool:
        verify_obj = pool.submit(compile_as, VERIFY_OBJECT, verify, output / "verify.cpp.o", overlay)
        # The selected build's only other verify.hpp consumer constructs Verifier instances.
        generate = overlay / GENERATE
        generate_obj = pool.submit(compile_as, GENERATE_OBJECT, generate if generate.is_file() else SOURCE / GENERATE,
                                   output / "generate.cpp.o", overlay)
        replacements = {}
        if include_qsa:
            qsa_obj = pool.submit(compile_as, QSA_OBJECT, overlay / QSA_KERNELS,
                                  output / "qsa_prompt_attn.cu.o", overlay)
        if cuda.exists():
            cuda_obj = pool.submit(compile_as, CUDA_OBJECT, cuda, output / "verify_kernels.cu.o", overlay)
            replacements["libstrata_kernels.a"] = replace_archive(
                output, cuda_obj.result(), "libstrata_kernels.a")
        if include_qsa:
            qsa_member = qsa_obj.result()
            if "libstrata_kernels.a" in replacements:
                subprocess.run(["ar", "r", str(replacements["libstrata_kernels.a"]), str(qsa_member)], check=True)
            else:
                replacements["libstrata_kernels.a"] = replace_archive(output, qsa_member, "libstrata_kernels.a")
        replacements["libstrata_engine.a"] = replace_archive(output, verify_obj.result(), "libstrata_engine.a")
        replacements[GENERATE_OBJECT] = generate_obj.result()
    link_as("strata", output / "strata", replacements)
    so = output / "libstrata_kq.so"
    if not so.exists():
        so.symlink_to(BUILD / so.name)


def operator(overlay, output):
    bench = overlay / "bench/q5-decode-tc"
    with ThreadPoolExecutor(max_workers=2) as pool:
        driver = pool.submit(compile_as, BENCH_OBJECT, bench / "bench.cpp", output / "bench.cpp.o", overlay)
        tensor = pool.submit(compile_as, CUDA_OBJECT, bench / "tensor_expert.cu",
                             output / "tensor_expert.cu.o", overlay)
        driver, tensor = driver.result(), tensor.result()
    link_as("native_expert_bench", output / "q5_decode_tc_bench", {BENCH_OBJECT: driver}, tensor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("capture", "operator", "qsa-ab"))
    parser.add_argument("overlay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    overlay, output = args.overlay.resolve(), args.output.resolve()
    if hashlib.sha256((BUILD / "strata").read_bytes()).hexdigest() != ENGINE_SHA:
        raise RuntimeError("The base build no longer matches the selected deployment")
    output.mkdir(parents=True, exist_ok=True)
    os.nice(5)
    if args.part == "operator":
        operator(overlay, output)
    else:
        capture(overlay, output, include_qsa=args.part == "qsa-ab")


if __name__ == "__main__":
    main()
