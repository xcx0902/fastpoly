#!/usr/bin/env python3
"""Regression checks for the single-header token compressor (stdlib only)."""

import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("amalgamate", ROOT / "scripts/amalgamate.py")
amalgamate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(amalgamate)


def tokens(source):
    return [token for token in amalgamate.tokenize(source) if token != "\n"]


class AmalgamateTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which(os.environ.get('CXX', 'clang++')), 'C++ preprocessor unavailable')
    def test_conditional_and_shared_dependencies(self):
        # A conditional include must not consume a dependency needed by another
        # arm or by a later unconditional include. Diamond dependencies still
        # define the shared type once, with its declaration before its users.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            headers = root / 'include/fastpoly'
            headers.mkdir(parents=True)
            sources = {
                'leaf': 'struct Leaf { int value; };\n',
                'adapter': '#include "fastpoly/leaf.hpp"\nstruct Adapter { Leaf value; };\n',
                'sample': ('#if defined(PICK_FIRST)\n#include "fastpoly/leaf.hpp"\n'
                           '#elif defined(PICK_SECOND)\n#include "fastpoly/leaf.hpp"\n#endif\n'
                           '#include "fastpoly/adapter.hpp"\n#include "fastpoly/leaf.hpp"\n'
                           'Adapter result;\n'),
            }
            for name, body in sources.items():
                guard = f'FASTPOLY_{name.upper()}_HPP'
                (headers / f'{name}.hpp').write_text(
                    f'#ifndef {guard}\n#define {guard}\n{body}\n#endif\n')
            with patch.object(amalgamate, 'ROOT', root):
                expanded = amalgamate.header_body('sample')
            base = [os.environ.get('CXX', 'clang++'), '-E', '-P', '-x', 'c++',
                    '-I', str(root / 'include')]
            for flags in ([], ['-DPICK_FIRST'], ['-DPICK_SECOND']):
                with self.subTest(flags=flags):
                    original = subprocess.check_output(base + flags + [str(headers / 'sample.hpp')], text=True)
                    bundled = subprocess.check_output(base + flags + ['-'], input=expanded, text=True)
                    self.assertEqual(tokens(original), tokens(bundled))
                    self.assertEqual(tokens(bundled).count('Leaf'), 2)

    def test_token_boundaries(self):
        source = """
        a + +b; a - -b; a / *b; a / /b; a < : b; a : : b;
        X<X<int> > v; X<::T> v; < :: :; . . .; 1 e+2; 0 xFF; 0x1.fp+2; 1'000ULL;
        u8 "a"; L 'x'; "a" _suffix; R"d(a\n// /* \" )d"; "// /* \\\"";
        """
        self.assertEqual(tokens(source), tokens(amalgamate.compact(source)))

    def test_comments_are_separators(self):
        self.assertEqual(amalgamate.compact("int/**/x; // end\nint y;"),
                         "int x;int y;\n")

    def test_literal_contents(self):
        source = 'auto a=u8R"tag( // /*\n\\\n)tag"; auto b="x\\\ny";'
        result = amalgamate.compact(source)
        self.assertIn('u8R"tag( // /*\n\\\n)tag"', result)
        self.assertIn('"xy"', result)
        self.assertEqual(tokens(source), tokens(result))

    def test_continued_comments_and_directives(self):
        source = '// hide\\\nint hidden;\n#define VALUE (1 + \\\n2)\nint x = VALUE;'
        self.assertEqual(amalgamate.compact(source), '#define VALUE (1+2)\nint x=VALUE;\n')

    def test_function_and_object_macro_spacing(self):
        source = '#define F(x) ((x) + 1)\n#define G (x)\n#if F(1)\nG\n#endif\n'
        result = amalgamate.compact(source)
        self.assertIn('#define F(', result)
        self.assertIn('#define G (', result)
        self.assertEqual(tokens(source), tokens(result))

    def test_invalid_literals_and_comments(self):
        for source in ('/*', '"unfinished', 'R"missing', 'R"x(unclosed'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                amalgamate.compact(source)

    def test_dictionary_expands_to_identical_tokens(self):
        # Plenty of repeated text makes factoring profitable after save/restore.
        block = tokens('inline constexpr uint32_t long_name(uint32_t x){return x+1;}')
        code = [block * 120, tokens('uint32_t unrelated;')]
        compressed, dictionary = amalgamate.compress(code)
        self.assertTrue(dictionary)
        definitions = dict(dictionary)

        def expand(block):
            result = []
            for token in block:
                result.extend(expand(definitions[token]) if token in definitions else [token])
            return result

        self.assertEqual([expand(block) for block in compressed], code)
        before = sum(len(amalgamate.join_tokens(block)) for block in code)
        after = (sum(len(amalgamate.join_tokens(block)) for block in compressed)
                 + sum(len(amalgamate.macro_open(name, pattern))
                       + len(amalgamate.macro_close(name)) for name, pattern in dictionary))
        self.assertLess(after, before)
        for block in compressed:
            self.assertEqual(tokens(amalgamate.join_tokens(block)), block)

    def test_function_macro_argument_boundaries(self):
        for source in ('(a,b)', 'a,b', 'a)', 'f(a', ');int x=0;'):
            self.assertFalse(amalgamate.safe_pattern(tuple(tokens(source))), source)
        for source in ('uint32_t', 'f(a,b)', 'const T x=f(a,b);'):
            self.assertTrue(amalgamate.safe_pattern(tuple(tokens(source))), source)

    @unittest.skipUnless(shutil.which(os.environ.get('CXX', 'clang++')), 'C++ preprocessor unavailable')
    def test_complete_header_preprocessor_equivalence(self):
        # Empty system headers isolate the library token stream. Compare every
        # backend, configuration overrides, repeated inclusion and macro hygiene.
        original = ('#ifndef FASTPOLY_SINGLE_HPP\n#define FASTPOLY_SINGLE_HPP\n'
                    + ''.join(amalgamate.header_body(name) for name in amalgamate.HEADERS)
                    + '\n#endif\n')
        generated = (ROOT / 'fastpoly.hpp').read_text()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for header in re.findall(r'#\s*include\s*<([^>]+)>', original):
                (directory / header).write_text('')
            # Real intrinsic headers contain function macros. Stub a few of
            # these too: empty headers alone cannot expose argument hiding.
            (directory / 'immintrin.h').write_text(
                '#define _mm256_extracti128_si256(V,M) fake_extract(V,M)\n'
                '#define _mm512_extracti64x4_epi64(V,M) fake_extract512(V,M)\n'
                '#define _mm256_permute2x128_si256(A,B,M) fake_perm(A,B,M)\n'
                '#define _mm512_inserti64x4(A,B,M) fake_insert(A,B,M)\n')
            compiler = os.environ.get('CXX', 'clang++')
            base = [compiler, '-std=c++20', '-E', '-P', '-x', 'c++', '-nostdinc',
                    '-nostdinc++', '-I', temporary]
            architecture_macros = ['__AVX512F__', '__AVX512VL__', '__AVX2__',
                                   '__ARM_NEON', '__ARM_NEON__', '_M_ARM64']
            base.extend('-U' + name for name in architecture_macros)
            variants = [[], ['-D__ARM_NEON=1'], ['-D__AVX2__=1'],
                        ['-D__AVX512F__=1', '-D__AVX512VL__=1']]
            prefix = '#define FP0 caller_value\n#define FP1(x) ((x)+7)\n'
            suffix = '\nFP0 FP1(2)\n#ifdef FP2\n#error leaked compression macro\n#endif\n'
            suffix += ('FASTPOLY_SCRATCH_CACHE_BYTES FPX_HAVE_AVX2_INTRIN '
                       'FPX_SIMD_AVX2 FPX_SIMD_AVX512 FPX_SIMD_NEON\n')
            for flags in variants:
                for cache in ('', '#define FASTPOLY_SCRATCH_CACHE_BYTES 0\n'):
                    with self.subTest(flags=flags, cache=cache):
                        outputs = []
                        for header in (original, generated):
                            process = subprocess.run(base + flags + ['-'],
                                                     input=prefix + cache + header + header + suffix,
                                                     text=True, capture_output=True, check=True)
                            outputs.append(tokens(process.stdout))
                        self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
