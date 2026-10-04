#!/usr/bin/env python3
"""Alternate two benchmark binaries and compare medians and full-output hashes."""

import argparse
import csv
import io
from pathlib import Path
import statistics
import subprocess
import sys


def measure(binary, args, op):
    command = [str(binary), str(args.max_size), "--min-size", str(args.min_size),
               "--op", op, "--reps", str(args.reps), "--warmup", str(args.warmup), "--csv"]
    rows = list(csv.DictReader(io.StringIO(subprocess.check_output(command, text=True))))
    return {(row["operation"], int(row["size"])): row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--min-size", type=int, default=1024)
    parser.add_argument("--max-size", type=int, default=1048576)
    parser.add_argument("--ops", default="all,square,skinny,exp-linear,inv-short,pow-short,pow-linear,div-small")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.reps < 1 or args.warmup < 0:
        parser.error("runs/reps must be positive; warmup must be nonnegative")
    samples = [{}, {}]
    for run in range(args.runs):
        for op in args.ops.split(","):
            for index in (0, 1) if run % 2 == 0 else (1, 0):
                binary = (args.before, args.after)[index].resolve()
                for key, row in measure(binary, args, op).items():
                    samples[index].setdefault(key, []).append(row)
            print(f"paired run {run + 1}/{args.runs}: {op}", file=sys.stderr, flush=True)
    if samples[0].keys() != samples[1].keys():
        raise RuntimeError("benchmark operations/sizes differ")
    output = []
    for key in sorted(samples[0], key=lambda item: (item[1], item[0])):
        before, after = (samples[index][key] for index in (0, 1))
        hashes = {row["checksum"] for row in before + after}
        if len(hashes) != 1:
            raise RuntimeError(f"output mismatch at {key}: {hashes}")
        backends = {row["backend"] for row in before + after}
        if len(backends) != 1:
            raise RuntimeError(f"different SIMD backends at {key}: {backends}")
        b = statistics.median(float(row["median_ms"]) for row in before)
        a = statistics.median(float(row["median_ms"]) for row in after)
        output.append([next(iter(backends)), key[0], key[1], f"{b:.9f}", f"{a:.9f}",
                       f"{b / a:.6f}", next(iter(hashes)), args.runs, args.reps, args.warmup])
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["backend", "operation", "size", "before_ms", "after_ms", "speedup",
                         "checksum", "runs", "reps", "warmup"])
        writer.writerows(output)
    print(f"{len(output)} pairs: all hashes match; wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
