#!/usr/bin/env python3
"""Select route records whose owning windows ran at each full decode width."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def select(capture, width):
    cohort = json.loads((capture / f"cohort{width}.json").read_bytes())
    if not cohort["qualified"]:
        raise ValueError(f"C{width} failed the full-width checks")
    begin, end = cohort["routes_roi_epoch_ns"]
    layers = Counter()
    windows_by_rows = Counter()
    experts_by_rows = Counter()
    records = []
    for route_file in sorted(capture.glob("routes.pid*.d*.routes")):
        inside = False
        for line in route_file.read_text().splitlines():
            if line.startswith("# window "):
                header = {key: int(value) for key, value in re.findall(r"([a-z_]+)=(\d+)", line)}
                inside = (header.get("tag") == width and begin <= header.get("epoch_us", 0) * 1000
                          <= header.get("done_us", 0) * 1000 <= end)
            elif inside and line and not line.startswith("#"):
                layer, rows, *ids = map(int, line.split())
                if not (0 <= layer < 48 and 1 <= rows <= 4 and len(ids) == rows * 10
                        and all(0 <= expert < 512 for expert in ids)
                        and all(len(set(ids[i:i + 10])) == 10 for i in range(0, len(ids), 10))):
                    raise ValueError(f"Unexpected routed rows in {route_file.name}")
                layers[layer] += 1
                windows_by_rows[rows] += 1
                experts_by_rows.update(Counter(ids).values())
                records.append(line)
    if len(layers) != 48 or min(layers.values()) < 8:
        raise ValueError(f"C{width} captured too few full-width windows per layer: {dict(layers)}")
    (capture / f"c{width}.routes").write_text("\n".join(records) + "\n")
    return {"records": len(records), "windows_by_rows": dict(sorted(windows_by_rows.items())),
            "experts_by_rows": dict(sorted(experts_by_rows.items())), "per_layer": dict(sorted(layers.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    report = {f"c{width}": select(args.capture, width) for width in (8, 16)}
    (args.capture / "route-shapes.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({arm: {k: v for k, v in row.items() if k != "per_layer"} for arm, row in report.items()}),
          flush=True)


if __name__ == "__main__":
    main()
