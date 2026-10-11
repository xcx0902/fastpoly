#!/usr/bin/env python3
"""Alternate tiny-NTT batch binaries; retain every process sample and checksum."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--sizes", default="2,4,8,16,32,64,128,256,512,1024")
    parser.add_argument("--ops", default="forward,inverse")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--reps", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = [int(n) for n in args.sizes.split(",")]
    ops = args.ops.split(",")
    if (min(args.runs, args.reps, args.elements) < 1 or args.warmup < 0 or
            any(n < 2 or n & (n - 1) for n in sizes) or
            any(op not in ("forward", "inverse", "prefix-quarter", "prefix-half", "highhalf") for op in ops)):
        parser.error("invalid batch size, operation or repetition count")
    binaries = [args.before.resolve(), args.after.resolve()]
    metadata = {"backend": args.backend, "binaries": [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in binaries], "samples": []}
    samples = {(op, n): [[], []] for n in sizes for op in ops}
    for run in range(args.runs):
        for n in sizes:
            for op in ops:
                for version in ((0, 1) if run % 2 == 0 else (1, 0)):
                    command = [str(binaries[version]), op, str(n),
                               str(max(1, args.elements // n)), str(args.reps), str(args.warmup)]
                    row = json.loads(subprocess.check_output(command, text=True))
                    samples[op, n][version].append(row)
                    metadata["samples"].append({"run": run + 1, "version": version,
                                                "command": command, **row})
            print(f"paired batch {run + 1}/{args.runs}: n={n}", flush=True)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["backend", "operation", "size", "before_ns", "after_ns", "speedup",
                         "checksum", "runs", "reps", "warmup", "loops"])
        for (op, n), pair in samples.items():
            hashes = {r["checksum"] for side in pair for r in side}
            if len(hashes) != 1:
                raise RuntimeError(f"checksum mismatch: {op} n={n}")
            before, after = [statistics.median(r["median_ns"] for r in side) for side in pair]
            writer.writerow([args.backend, op, n, f"{before:.9f}", f"{after:.9f}",
                             f"{before / after:.6f}", hashes.pop(), args.runs, args.reps,
                             args.warmup, max(1, args.elements // n)])
    print(f"{len(samples)} pairs: all hashes match; wrote {args.output}")


if __name__ == "__main__":
    main()
