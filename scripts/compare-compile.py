#!/usr/bin/env python3
"""Alternate clean -O2 compilations against two header directories (no cache)."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent
PROBES = {
    "include": "",
    "ntt": """
void transform(uint32_t* a, uint32_t n) {
  const auto& plan = fpx::NttPlan<M>::get_ref(n);
  plan.forward(a);
  plan.inverse(a);
}
""",
    "conv": "V product(const V& a, const V& b, size_t n) { return fpx::poly::conv(a, b, n); }",
    "series": """
V reciprocal(const V& a, size_t n) { return fpx::poly::inv(a, n); }
V logarithm(const V& a, size_t n) { return fpx::poly::log(a, n); }
V exponential(const V& a, size_t n) { return fpx::poly::exp(a, n); }
V square_root(const V& a, size_t n) { return fpx::poly::sqrt(a, n); }
V power(const V& a, uint64_t k, size_t n) { return fpx::poly::pow(a, k, n); }
""",
}
SOURCES = {
    "test_ntt": ROOT / "tests/test_ntt.cpp",
    "test_poly": ROOT / "tests/test_poly.cpp",
    "test_prime_poly": ROOT / "tests/test_prime_poly.cpp",
    "bench": ROOT / "bench/bench.cpp",
}


def header_hash(directory, header):
    digest = hashlib.sha256()
    # A standalone header's parent may contain unrelated benchmark artifacts.
    # Fingerprint only the distribution being compiled.
    paths = (sorted((directory / "fastpoly").rglob("*.hpp"))
             if header == "fastpoly/fastpoly.hpp" else [directory / header])
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True, help="baseline include directory")
    parser.add_argument("--after", type=Path, default=ROOT / "include", help="candidate include directory")
    parser.add_argument("--cxx", default="clang++", help="compiler executable, bypassing ccache/sccache")
    parser.add_argument("--flag", action="append", default=[], help="extra flag; use --flag=-mavx2")
    parser.add_argument("--header", default="fastpoly/fastpoly.hpp", help="probe header, e.g. fastpoly.hpp")
    parser.add_argument("--modulus", type=int, default=998244353, help="field modulus for generated probes")
    parser.add_argument("--root", type=int, help="probe primitive root; defaults to 3 for 998244353, otherwise automatic")
    parser.add_argument("--cases", default=",".join([*PROBES, *SOURCES]))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True, help="CSV; raw samples go to a JSON sidecar")
    args = parser.parse_args()
    if not 2 <= args.modulus < 2**31:
        parser.error("modulus must be a prime in [2, 2^31)")
    root = args.root if args.root is not None else (3 if args.modulus == 998244353 else 0)
    if not 0 <= root < args.modulus:
        parser.error("root must be zero (automatic) or less than the modulus")
    cases = args.cases.split(",")
    if args.runs < 1 or not cases or any(case not in PROBES and case not in SOURCES for case in cases):
        parser.error("runs must be positive and cases must name known probes or sources")
    if len(set(cases)) != len(cases):
        parser.error("cases must not contain duplicates")
    if args.header != "fastpoly/fastpoly.hpp" and any(case not in ("include", "conv", "series") for case in cases):
        parser.error("custom headers support only include,conv,series (public polynomial APIs)")
    directories = [args.before.resolve(), args.after.resolve()]
    for directory in directories:
        if not (directory / args.header).is_file():
            parser.error(f"header not found: {directory / args.header}")
    flags = ["-std=c++20", "-O2", "-DNDEBUG", "-pthread", *args.flag]
    metadata = {
        "compiler": subprocess.check_output([args.cxx, "--version"], text=True),
        "flags": flags,
        "header": args.header,
        "probe_modulus": args.modulus,
        "probe_root": root,
        "directories": list(map(str, directories)),
        "header_sha256": [header_hash(directory, args.header) for directory in directories],
        "sources": {},
        "samples": [],
    }
    samples = {case: [[], []] for case in cases}
    cpu_samples = {case: [[], []] for case in cases}
    sizes = {case: [0, 0] for case in cases}
    with tempfile.TemporaryDirectory(prefix="fastpoly-compile-") as temporary:
        work = Path(temporary)
        sources = {}
        for case in cases:
            if case in PROBES:
                source = work / (case + ".cpp")
                source.write_text(f'#include "{args.header}"\n'
                                  f'using M = fpx::Mont<{args.modulus}u, {root}u>;\n'
                                  'using V = std::vector<M>;\n' + PROBES[case] + '\n')
            else:
                source = SOURCES[case]
            sources[case] = source
            content = source.read_bytes()
            metadata["sources"][case] = {"sha256": hashlib.sha256(content).hexdigest()}
            if case in PROBES:
                metadata["sources"][case]["probe"] = content.decode()
            else:
                metadata["sources"][case]["path"] = str(source.relative_to(ROOT))
        for run in range(args.runs):
            for case in cases:
                for index in (0, 1) if run % 2 == 0 else (1, 0):
                    output = work / f"{case}-{index}.o"
                    command = [args.cxx, *flags, "-I" + str(directories[index]),
                               "-c", str(sources[case]), "-o", str(output)]
                    cpu_start = os.times()
                    start = time.perf_counter()
                    subprocess.run(command, check=True)
                    seconds = time.perf_counter() - start
                    cpu_end = os.times()
                    cpu_seconds = (cpu_end.children_user + cpu_end.children_system
                                   - cpu_start.children_user - cpu_start.children_system)
                    samples[case][index].append(seconds)
                    cpu_samples[case][index].append(cpu_seconds)
                    sizes[case][index] = output.stat().st_size
                    metadata["samples"].append({"case": case, "run": run + 1, "version": index,
                                                "seconds": seconds, "cpu_seconds": cpu_seconds,
                                                "object_bytes": sizes[case][index],
                                                "command": shlex.join(command)})
                print(f"paired compile {run + 1}/{args.runs}: {case}", file=sys.stderr, flush=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "before_s", "after_s", "speedup", "before_bytes", "after_bytes", "runs",
                         "before_cpu_s", "after_cpu_s", "cpu_speedup"])
        for case in cases:
            before, after = map(statistics.median, samples[case])
            before_cpu, after_cpu = map(statistics.median, cpu_samples[case])
            # Some platforms do not expose child CPU accounting.
            cpu_speedup = f"{before_cpu/after_cpu:.6f}" if before_cpu > 0 and after_cpu > 0 else ""
            writer.writerow([case, f"{before:.6f}", f"{after:.6f}", f"{before/after:.6f}",
                             *sizes[case], args.runs, f"{before_cpu:.6f}", f"{after_cpu:.6f}", cpu_speedup])
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{len(cases)} cases: wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
