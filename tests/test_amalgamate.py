#!/usr/bin/env python3
"""Regression checks for the single-header token compressor (stdlib only)."""

import importlib.util
from itertools import combinations
import io
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


def prepare_single_header_tests(directory):
    """Translate the suite's implementation references, keeping public APIs.

    Use the same identifier map as the default bundle. No aliases are added to
    the distributed header merely to support tests of implementation details.
    """
    directory.mkdir(parents=True, exist_ok=True)
    mapping = amalgamate.implementation_renames(amalgamate.chunks(amalgamate.header_body('fastpoly')))
    shutil.copy(ROOT / 'tests/fp_test.hpp', directory / 'fp_test.hpp')
    for source in (ROOT / 'tests').glob('test_*.cpp'):
        parts = amalgamate.chunks(source.read_text())
        # Tests have their own local names (e.g. f or D) which may coincide with
        # the bundle's new names. Move those aside before translating references.
        occupied = {t for _, block in parts for t in block}
        collisions = {}
        for short in sorted(set(mapping.values()) & occupied):
            fresh = 'fpx_test_original_' + short
            while fresh in occupied:
                fresh += '_'
            collisions[short] = fresh
            occupied.add(fresh)
        parts = amalgamate.rename_parts(parts, collisions)
        # Atomic load/store and explicitly qualified standard-library calls in
        # the tests are external, even when a SIMD helper shares the spelling.
        parts = [(directive, block if directive else [
            token if ((i >= 2 and block[i - 2:i] == ['std', '::'])
                      or (i and block[i - 1] in ('.', '->') and token in ('load', 'store')))
            else mapping.get(token, token)
            for i, token in enumerate(block)]) for directive, block in parts]
        (directory / source.name).write_text(''.join(
            amalgamate.directive(block) if is_directive else amalgamate.join_tokens(block) + '\n'
            for is_directive, block in parts))


class AmalgamateTests(unittest.TestCase):
    def test_command_line_paths_and_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'selected.hpp'
            argv = ['amalgamate.py', '--paths', 'scalar', 'neon', '--output', str(output)]
            with patch.object(amalgamate, 'generate', return_value='selected header\n') as generate:
                with patch.object(sys, 'argv', argv):
                    self.assertEqual(amalgamate.main(), 0)
                generate.assert_called_once_with(['scalar', 'neon'])
                self.assertEqual(output.read_text(), 'selected header\n')
                with patch.object(sys, 'argv', argv + ['--check']):
                    self.assertEqual(amalgamate.main(), 0)
                    output.write_text('stale\n')
                    with patch.object(sys, 'stderr', io.StringIO()):
                        self.assertEqual(amalgamate.main(), 1)
                    self.assertEqual(output.read_text(), 'stale\n')
            for args in (['--paths'], ['--paths', 'unknown']):
                with patch.object(sys, 'argv', ['amalgamate.py', *args]), \
                     patch.object(sys, 'stderr', io.StringIO()), \
                     self.assertRaises(SystemExit) as error:
                    amalgamate.main()
                self.assertEqual(error.exception.code, 2)

    def test_implementation_names_and_public_collisions(self):
        source = '''
        #define CONFIG internal_macro
        namespace fpx {
        int shared_name(int value) { return value; }
        namespace detail {
        int long_helper(int long_variable) {
          auto f0 = std::move(long_variable);
          return shared_name(f0) + object.size() + CONFIG;
        }
        struct LongBuffer {
          LongBuffer() = default;
          void reset() { long_helper(0); }
        };
        }
        int public_function(int value) { return detail::long_helper(value); }
        class Poly {
        public:
          int degree() const { int polynomial_degree = 4; return polynomial_degree; }
        private:
          int coefficient_storage;
        };
        }
        namespace fpx::poly::detail {
        int other_helper() { return fpx::detail::long_helper(1); }
        const char* literal = "long_helper long_variable";
        }
        '''
        parts = amalgamate.chunks(source)
        mapping = amalgamate.implementation_renames(parts)
        for name in ('long_helper', 'long_variable', 'other_helper', 'LongBuffer',
                     'public_function', 'shared_name', 'polynomial_degree', 'coefficient_storage'):
            self.assertIn(name, mapping)
            self.assertLess(len(mapping[name]), len(name))
        for name in ('Poly', 'degree', 'move', 'size',
                     'CONFIG', 'internal_macro', 'reset', 'int', 'return'):
            self.assertNotIn(name, mapping)
        self.assertNotIn('f0', mapping.values())
        self.assertEqual(len(set(mapping.values())), len(mapping))
        renamed = amalgamate.rename_parts(parts, mapping)
        self.assertIn('"long_helper long_variable"', [t for _, b in renamed for t in b])
        reverse = {short: name for name, short in mapping.items()}
        self.assertEqual(amalgamate.rename_parts(renamed, reverse), parts)

    def test_full_implementation_renames(self):
        parts = amalgamate.chunks(amalgamate.header_body('fastpoly'))
        mapping = amalgamate.implementation_renames(parts)
        # Namespace, SIMD, NTT, public-method locals, private members and
        # template parameters all participate, beyond the old detail-only pass.
        for name in ('detail', 'simd', 'NttPlan', 'forward_prefix', 'mul_twiddle',
                     'data_', 'stage_count_', 'Mod', 'factors', 'remaining',
                     'from_int_impl', 'scratch_memory_stats', 'root', 'inner_n'):
            self.assertIn(name, mapping)
            self.assertLess(len(mapping[name]), len(name))
        self.assertTrue(amalgamate.PUBLIC_NAMES.isdisjoint(mapping))
        for name in ('uint32x4_t', 'vaddq_u32', 'vld1q_u32', '_mm256_mul_epu32',
                     'memcpy', 'data', 'digits', 'size'):
            self.assertNotIn(name, mapping)
        self.assertEqual(len(mapping), len(set(mapping.values())))
        existing = {t for _, block in parts for t in block}
        self.assertTrue(existing.isdisjoint(mapping.values()))
        self.assertEqual(mapping, amalgamate.implementation_renames(parts))

    def test_short_names(self):
        names = [amalgamate.short_name(i) for i in range(5000)]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(amalgamate.IDENTIFIER.fullmatch(n) for n in names))
        self.assertEqual(list(map(len, names)), sorted(map(len, names)))
        self.assertEqual(names[0], 'a')
        self.assertEqual(names[51], 'Z')
        self.assertEqual(names[52], 'aa')

    def test_path_subsets(self):
        for count in range(1, 5):
            for paths in combinations(amalgamate.PATHS, count):
                with self.subTest(paths=paths):
                    source = amalgamate.header_body('fastpoly', paths)
                    for path in amalgamate.PATHS:
                        guard = f'FASTPOLY_DETAIL_SIMD_{path.upper()}_HPP'
                        self.assertEqual(guard in source, path in paths)
                    self.assertEqual('#include <arm_neon.h>' in source, 'neon' in paths)
                    self.assertEqual('#include <immintrin.h>' in source,
                                     bool({'avx2', 'avx512'} & set(paths)))
                    self.assertEqual('no enabled SIMD path' in source, 'scalar' not in paths)
        for paths in ([], ['bogus'], ['scalar', 'bogus']):
            with self.assertRaises(ValueError):
                amalgamate.generate(paths)
        self.assertEqual(amalgamate.normalize_paths(['avx2', 'scalar', 'avx2']),
                         ('scalar', 'avx2'))

    @unittest.skipUnless(shutil.which(os.environ.get('CXX', 'clang++')), 'C++ compiler unavailable')
    def test_public_api_compile_run(self):
        source = '''
        #include "selected.hpp"
        #include <cassert>
        #include <sstream>
        int main() {
          using M = fpx::ModInt<998244353>;
          using P = fpx::Poly<M>;
          static_assert(M::mod == 998244353 && M(7).val() == 7);
          P a{1, 2, 3}, b{4, 5};
          assert((a*b).to_ints() == std::vector<uint32_t>({4, 13, 22, 15}));
          a += b; a -= b; assert(a == P({1, 2, 3}));
          P c(129, M(1));
          auto product = c.mul(c.inv(129), 129);
          assert(product[0] == M(1));
          for (size_t i = 1; i < product.size(); ++i) assert(product[i] == M());
          assert(c.log(129).exp(129) == c);
          assert(c.pow(2, 129) == c.mul(c, 129));
          std::stringstream stream; stream << M(42);
          M value; stream >> value; assert(value.val() == 42);
          using N = fpx::ModInt<1000000007>;
          fpx::Poly<N> d(100, N(1));
          auto e = d*d; assert(e[99] == N(100));
        }
        '''
        # The checked-in bundle exercises dictionary compression. For scalar,
        # isolate filtering and renaming without rebuilding that dictionary.
        with patch.object(amalgamate, 'compress', side_effect=lambda blocks: (blocks, [])):
            scalar = amalgamate.generate(['scalar'])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'main.cpp').write_text(source)
            for label, header in (('all', (ROOT / 'fastpoly.hpp').read_text()), ('scalar', scalar)):
                with self.subTest(paths=label):
                    (directory / 'selected.hpp').write_text(header)
                    result = subprocess.run(
                        [os.environ.get('CXX', 'clang++'), '-std=c++20', '-O2',
                         str(directory / 'main.cpp'), '-o', str(directory / 'test')],
                        text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    subprocess.run([str(directory / 'test')], check=True, capture_output=True)

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
        mapping = amalgamate.implementation_renames(amalgamate.chunks(original))
        reverse = {short: name for name, short in mapping.items()}
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
                        self.assertEqual(outputs[0], [reverse.get(t, t) for t in outputs[1]])

    @unittest.skipUnless(shutil.which(os.environ.get('CXX', 'clang++')), 'C++ preprocessor unavailable')
    def test_subset_dispatch(self):
        # Every subset against every target, including fallback and failure.
        source = (ROOT / 'include/fastpoly/simd.hpp').read_text()
        selection = source[source.index('#if defined(__AVX512F__)'):source.index('namespace fpx::simd {', source.index('#if defined(__AVX512F__)'))]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'fastpoly/detail/simd').mkdir(parents=True)
            for path in amalgamate.PATHS:
                (directory / f'fastpoly/detail/simd/{path}.hpp').write_text(f'CHOSEN_{path}\n')
            for header in ('arm_neon.h', 'immintrin.h'):
                (directory / header).write_text('')
            macros = ('__AVX512F__', '__AVX512VL__', '__AVX2__', '__ARM_NEON', '__ARM_NEON__', '_M_ARM64')
            base = [os.environ.get('CXX', 'clang++'), '-E', '-P', '-x', 'c++', '-nostdinc',
                    '-I', temporary, *('-U' + m for m in macros)]
            targets = [([], []), (['__ARM_NEON'], ['neon']),
                       (['__AVX2__'], ['avx2']),
                       (['__AVX512F__', '__AVX512VL__', '__AVX2__'], ['avx512', 'avx2']),
                       (['__AVX512F__', '__AVX2__'], ['avx2'])]
            for count in range(1, 5):
                for paths in combinations(amalgamate.PATHS, count):
                    selected = amalgamate.select_paths(selection, paths)
                    for flags, supported in targets:
                        expected = next((p for p in [*supported, 'scalar'] if p in paths), None)
                        with self.subTest(paths=paths, flags=flags):
                            result = subprocess.run(base + ['-D' + f for f in flags] + ['-'],
                                                    input=selected, text=True, capture_output=True)
                            if expected:
                                self.assertEqual(result.returncode, 0, result.stderr)
                                self.assertEqual(result.stdout.strip(), 'CHOSEN_' + expected)
                            else:
                                self.assertNotEqual(result.returncode, 0)
                                self.assertIn('no enabled SIMD path', result.stderr)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--prepare-single-header-tests':
        prepare_single_header_tests(Path(sys.argv[2]))
    else:
        unittest.main()
