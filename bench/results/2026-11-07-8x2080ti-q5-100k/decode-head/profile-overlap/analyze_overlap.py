#!/usr/bin/env python3
"""Read an existing Nsight SQLite export; this script never invokes CUDA or captures a profile.

Reproduce the saved result from the local checkout:
  python3 analyze_overlap.py --ssh root@10.100.3.3 --output overlap.json

Or run beside a copied database:
  python3 analyze_overlap.py --database /path/to/decode.sqlite --output overlap.json

Head identification is specific to this Q6_K output-head geometry. Kernel intervals use Nsight's
nanosecond timeline. A union counts concurrent kernels once; intersections are observations,
not a dependency-graph reconstruction or a prediction of offload performance.
"""

import argparse
import bisect
import collections
import hashlib
import json
import pathlib
import shlex
import sqlite3
import statistics
import subprocess
import sys


DEFAULT_DATABASE = "/srv/bench100k-20261107-candidates/profile-groups8/decode.sqlite"


def interval_union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if end > start:
            result.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def overlap_record(subject, other):
    total = duration(subject)
    overlapping = duration(intersection(subject, other))
    return {
        "overlap_ns": overlapping,
        "overlap_ms": overlapping / 1e6,
        "fraction_of_subject": overlapping / total if total else None,
        "subject_exclusive_of_other_ms": (total - overlapping) / 1e6,
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(args):
    path = pathlib.Path(args.database).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        span_start, span_end, all_count = connection.execute(
            "SELECT min(start), max(end), count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
        rows = list(connection.execute(
            "SELECT k.start, k.end, k.streamId, k.gridX, s.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
            "WHERE k.deviceId=? ORDER BY k.start", (args.source_gpu,)
        ))
        # Q6_K T1: one CTA/output row. IL: 4, 8 or 16 output rows/CTA according to rows/warp.
        il_grids = {(args.vocab + rows_per_cta - 1) // rows_per_cta for rows_per_cta in (4, 8, 16)}

        def is_main(row):
            grid, name = row[3], row[4]
            return ("IlQ6K" in name and grid in il_grids) or (
                "native_q6_k_mmvq_kernel" in name and grid == args.vocab
            )

        def is_draft(row):
            return row[3] == args.draft_vocab and "native_q6_k_mmvq_kernel" in row[4]

        main_rows = [row for row in rows if is_main(row)]
        draft_rows = [row for row in rows if is_draft(row)]
        if not main_rows or not draft_rows:
            raise RuntimeError("No matching main/draft head kernels; check device, geometry and kernel names")
        main_streams = sorted({row[2] for row in main_rows})
        draft_streams = sorted({row[2] for row in draft_rows})
        intervals = lambda selected: interval_union((row[0], row[1]) for row in selected)
        main = intervals(main_rows)
        draft = intervals(draft_rows)
        draft_stream = intervals(row for row in rows if row[2] in draft_streams)
        main_stream = intervals(row for row in rows if row[2] in main_streams)
        source_all = intervals(rows)
        non_main_head = intervals(row for row in rows if not is_main(row))
        non_draft_stream = intervals(row for row in rows if row[2] not in draft_streams)
        targets = []
        for device in args.target_gpus:
            busy = interval_union(connection.execute(
                "SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=? ORDER BY start", (device,)
            ))
            targets.append({
                "device": device,
                "kernel_busy_union_ms": duration(busy) / 1e6,
                "kernel_busy_fraction_of_capture_span": duration(busy) / (span_end - span_start),
                **overlap_record(main, busy),
            })
        next_draft_starts = [row[0] for row in rows if row[2] in draft_streams]
        gaps = []
        for _, end in main:
            index = bisect.bisect_left(next_draft_starts, end)
            if index < len(next_draft_starts):
                gaps.append((next_draft_starts[index] - end) / 1e3)
        gaps.sort()
        stream_stats = collections.defaultdict(lambda: {"kernel_count": 0, "kernel_sum_ns": 0})
        for start, end, stream, _, _ in rows:
            stream_stats[stream]["kernel_count"] += 1
            stream_stats[stream]["kernel_sum_ns"] += end - start
        draft_sum = sum(row[1] - row[0] for row in draft_rows)
        weight_bytes = args.draft_vocab * (args.hidden // 256) * 210
        return {
            "schema_version": 1,
            "input": {
                "database": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path),
                "profile_kind": "single request, approximately 100k context, PW2; not C6/C8 batch",
                "source_gpu": args.source_gpu, "head_type": "Q6_K", "hidden": args.hidden,
                "main_vocab": args.vocab, "draft_vocab": args.draft_vocab,
            },
            "identification": {
                "main_single_column": f"native_q6_k_mmvq_kernel, gridX={args.vocab}",
                "main_interleaved": {"name_contains": "IlQ6K", "gridX": sorted(il_grids)},
                "draft": f"native_q6_k_mmvq_kernel, gridX={args.draft_vocab}",
                "main_stream_ids": main_streams, "draft_stream_ids": draft_streams,
                "draft_stream_scope": "All kernels on streams identified by the draft-subset head, including its non-head work",
            },
            "capture": {
                "kernel_event_start_ns": span_start, "kernel_event_end_ns": span_end,
                "kernel_event_span_ms": (span_end - span_start) / 1e6, "all_gpu_kernel_count": all_count,
            },
            "main_head": {"count": len(main_rows), "kernel_sum_ms": sum(row[1] - row[0] for row in main_rows) / 1e6,
                          "kernel_union_ms": duration(main) / 1e6},
            "draft_head": {"count": len(draft_rows), "kernel_sum_ms": draft_sum / 1e6,
                           "kernel_union_ms": duration(draft) / 1e6,
                           "logical_weight_bytes_per_call": weight_bytes,
                           "effective_weight_GB_per_second": weight_bytes * len(draft_rows) / draft_sum,
                           "effective_bandwidth_note": "Logical matrix bytes per call divided by kernel duration; not a DRAM counter measurement"},
            "source_gpu_unions_ms": {
                "all_kernels": duration(source_all) / 1e6,
                "main_head_stream_all_kernels": duration(main_stream) / 1e6,
                "draft_head_stream_all_kernels": duration(draft_stream) / 1e6,
            },
            "overlap": {
                "main_head_vs_draft_head": overlap_record(main, draft),
                "main_head_vs_draft_stream_all_kernels": overlap_record(main, draft_stream),
                "main_head_vs_source_gpu_all_other_kernels": overlap_record(main, non_main_head),
                "draft_head_vs_source_gpu_non_draft_stream_kernels": overlap_record(draft, non_draft_stream),
            },
            "target_gpu_activity_during_main_head": targets,
            "main_head_end_to_next_draft_stream_kernel_us": {
                "count": len(gaps), "min": min(gaps), "median": statistics.median(gaps),
                "p10_nearest_rank_index": gaps[int(len(gaps) * .1)],
                "p90_nearest_rank_index": gaps[int(len(gaps) * .9)], "max": max(gaps),
            },
            "source_stream_kernel_sums": [{"stream": stream, **values} for stream, values in sorted(stream_stats.items())],
            "limitations": [
                "CUDA graph node tracing overhead is included; these are not baseline decode timings.",
                "Kernel duration sums and busy unions do not reconstruct the critical path or measure SM/bandwidth utilization.",
                "Busy unions exclude copy-engine activity and CPU work.",
                "Zero overlap is an observation of this schedule, not proof that kernels can never overlap.",
                "Target occupancy is measured before offload; placing a head there changes the schedule.",
                "The main-head duration is not a speedup estimate: moving identical work to an identical GPU retains its dependency latency.",
                "Head identification assumes the stated Q6_K geometry and the observed native/IL dispatch layouts.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--source-gpu", type=int, default=7)
    parser.add_argument("--target-gpus", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--draft-vocab", type=int, default=106299)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--ssh", help="Read SQLite on this SSH host; only this Python source is sent over stdin")
    parser.add_argument("--output", type=pathlib.Path, help="Local JSON output; stdout when omitted")
    args = parser.parse_args()
    if args.hidden <= 0 or args.hidden % 256:
        parser.error("--hidden must be a positive multiple of 256 for Q6_K")
    if args.ssh:
        remote_args = ["python3", "-", "--database", args.database, "--source-gpu", str(args.source_gpu),
                       "--vocab", str(args.vocab), "--draft-vocab", str(args.draft_vocab), "--hidden", str(args.hidden),
                       "--target-gpus", *(str(gpu) for gpu in args.target_gpus)]
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.ssh, shlex.join(remote_args)],
            input=pathlib.Path(__file__).read_text(), text=True, capture_output=True, check=True,
        )
        report = json.loads(result.stdout)
        report["input"]["ssh_host"] = args.ssh
    else:
        report = analyze(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(str(args.output))
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
