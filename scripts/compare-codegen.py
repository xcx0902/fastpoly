#!/usr/bin/env python3
"""Compare complete optimized benchmark assembly across two header trees.

Only C++ symbol names are canonicalized: the selected backend's extra template
argument is removed from demangled names. Instructions, registers, constants,
branches, local labels, alignment and unwind data must otherwise match exactly.
This complements timing on machines without native hardware for every backend.
"""

import argparse
import csv
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
SYMBOL = re.compile(r"(?<!\w)_{1,2}Z[\w$]+")


def canonicalize(assembly: str, demangler: str) -> str:
    symbols = sorted(set(SYMBOL.findall(assembly)))
    if not symbols:
        return assembly
    # Mach-O prepends a C underscore to the Itanium C++ mangled symbol.
    names = [symbol[1:] if symbol.startswith('__Z') else symbol for symbol in symbols]
    suffixes = ['$tlv$init' if name.endswith('$tlv$init') else '' for name in names]
    names = [name.removesuffix(suffix) if suffix else name for name, suffix in zip(names, suffixes)]
    result = subprocess.check_output([demangler], input='\n'.join(names) + '\n', text=True)
    demangled = result.splitlines()
    if len(demangled) != len(symbols) or any(name.startswith('_Z') for name in demangled):
        raise RuntimeError('could not demangle every C++ symbol')
    aliases = {symbol: name.replace(', fpx::simd::backend', '') + suffix
               for symbol, name, suffix in zip(symbols, demangled, suffixes)}
    return SYMBOL.sub(lambda match: aliases[match[0]], assembly)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-include', type=Path, required=True)
    parser.add_argument('--after-include', type=Path, default=ROOT / 'include')
    parser.add_argument('--source', type=Path, default=ROOT / 'bench/bench.cpp')
    parser.add_argument('--cxx', default='clang++')
    parser.add_argument('--demangler', default='c++filt')
    parser.add_argument('--flag', action='append', default=[])
    parser.add_argument('--label', required=True)
    parser.add_argument('--moduli', default='998244353:3,1004535809:3,469762049:3,167772161:3,754974721:11,1224736769:3')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    failed = []
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for modulus in args.moduli.split(','):
            mod, root = map(int, modulus.split(':'))
            outputs = []
            for side, include in [('before', args.before_include), ('after', args.after_include)]:
                target = directory / f'{side}.s'
                command = [args.cxx, '-std=c++20', '-O2', '-DNDEBUG', *args.flag,
                           f'-DFASTPOLY_BENCH_MOD={mod}u', f'-DFASTPOLY_BENCH_ROOT={root}u',
                           '-I', str(include.resolve()), '-S', str(args.source.resolve()), '-o', str(target)]
                subprocess.run(command, check=True)
                outputs.append(canonicalize(target.read_text(), args.demangler))
            hashes = [hashlib.sha256(output.encode()).hexdigest() for output in outputs]
            identical = outputs[0] == outputs[1]
            rows.append([args.label, mod, root, args.source.name, *hashes, str(identical).lower()])
            print(f'{args.label} mod={mod}: {"identical" if identical else "DIFFERS"}', flush=True)
            if not identical:
                failed.append(mod)
                # Retain the evidence for inspection instead of hiding a mismatch.
                for side, output in zip(('before', 'after'), outputs):
                    args.output.with_name(f'{args.output.stem}-{mod}-{side}.s').write_text(output)
    with args.output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['backend', 'modulus', 'root', 'source', 'before_sha256', 'after_sha256', 'identical'])
        writer.writerows(rows)
    if failed:
        raise SystemExit(f'assembly changed for moduli: {failed}')


if __name__ == '__main__':
    main()
