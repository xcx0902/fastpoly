#!/usr/bin/env python3
"""Generate a compact standalone distribution from the public headers."""

import argparse
from bisect import bisect_left
from collections import Counter
from functools import lru_cache
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent.parent
# Follow the umbrella's actual dependency graph instead of a second manual order.
HEADERS = ("fastpoly",)
PATHS = ("scalar", "neon", "avx2", "avx512")
IDENTIFIER = re.compile(r"(?:[^\W\d]|_)\w*")
NUMBER = re.compile(r"(?:\d|\.\d)(?:[eEpP][+-]|[\w.]|'[\w])*", re.UNICODE)
LITERAL = re.compile(r'(?:u8|u|U|L)?(R"|"|\')')
PUNCTUATION = re.compile(
    r"%:%:|>>=|<<=|<=>|->\*|\.\.\.|##|::|\.\*|->|\+\+|--|<<|>>|"
    r"<=|>=|==|!=|&&|\|\||\*=|/=|%=|\+=|-=|&=|\^=|\|=|<:|:>|<%|%>|%:|.",
    re.DOTALL,
)


def tokenize(source: str, *, preserve_defines: bool = False) -> list[str]:
    """C++ preprocessing tokens, with physical newlines outside literals.

    Splice continuations before recognizing comments (translation phase 2).
    Literal contents, prefixes, suffixes and digit separators stay verbatim.
    """
    original = source
    positions = None
    continuations = list(re.finditer(r"\\\r?\n", source))
    if continuations:
        positions = []
        pieces = []
        start = 0
        for match in continuations:
            pieces.append(source[start:match.start()])
            positions.extend(range(start, match.start()))
            start = match.end()
        pieces.append(source[start:])
        positions.extend(range(start, len(source)))
        source = "".join(pieces)
    tokens = []
    i = 0
    while i < len(source):
        if source[i].isspace():
            if source[i] == "\n":
                tokens.append("\n")
            i += 1
        elif source.startswith("//", i):
            end = source.find("\n", i)
            i = len(source) if end == -1 else end
        elif source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end == -1:
                raise ValueError("unterminated block comment")
            tokens.extend(["\n"] * source[i:end + 2].count("\n"))
            i = end + 2
        elif match := NUMBER.match(source, i):
            tokens.append(match[0])
            i = match.end()
        elif match := LITERAL.match(source, i):
            if match[1] == 'R"':
                opening = source.find("(", match.end())
                delimiter = source[match.end():opening]
                if opening == -1 or len(delimiter) > 16 or re.search(r'[\s()\\]', delimiter):
                    raise ValueError("invalid raw string delimiter")
                closing = ")" + delimiter + '"'
                # Phase-2 splicing is reverted inside raw string contents.
                raw_source = original if positions is not None else source
                raw_opening = positions[opening] if positions is not None else opening
                end = raw_source.find(closing, raw_opening + 1)
                if end == -1:
                    raise ValueError("unterminated raw string literal")
                end += len(closing)
                if positions is not None:
                    end = bisect_left(positions, end)
            else:
                quote = match[1]
                end = match.end()
                while end < len(source):
                    if source[end] == "\\":
                        end += 2
                    elif source[end] == quote:
                        end += 1
                        break
                    else:
                        end += 1
                else:
                    raise ValueError("unterminated quoted literal")
            suffix = IDENTIFIER.match(source, end)
            if suffix:
                end = suffix.end()
            if match[1] == 'R"' and positions is not None:
                tokens.append(original[positions[i]:positions[end - 1] + 1])
            else:
                tokens.append(source[i:end])
            i = end
        elif match := IDENTIFIER.match(source, i):
            tokens.append(match[0])
            i = match.end()
            if (preserve_defines and len(tokens) >= 3
                    and tokens[-3] in ("#", "%:") and tokens[-2] == "define"
                    and source[i:i + 1] == "("):
                # Encode the significant lack of whitespace in '#define F('.
                tokens[-1] += "("
                i += 1
        elif source.startswith("<::", i) and source[i + 3:i + 4] not in (":", ">"):
            # C++ maximal-munch exception for '<::name' (not the '<:' digraph).
            tokens.append("<")
            i += 1
        else:
            match = PUNCTUATION.match(source, i)
            tokens.append(match[0])
            i = match.end()
    return tokens


@lru_cache(maxsize=None)
def needs_space(left: str, right: str) -> bool:
    # Retokenizing the boundary also catches comments, digraphs, pp-numbers,
    # literal prefixes/suffixes and operators such as '+ +' becoming '++'.
    if left == right == "." or (left, right) == ("<", "::"):
        return True  # Multi-token ellipsis/digraph ambiguities.
    try:
        return tokenize(left + right) != [left, right]
    except ValueError:
        return True


def join_tokens(tokens: list[str] | tuple[str, ...]) -> str:
    return "".join((" " if i and needs_space(tokens[i - 1], token) else "") + token
                   for i, token in enumerate(tokens))


def chunks(source: str) -> list[tuple[bool, list[str]]]:
    """Separate directive lines from code; never compress across a directive."""
    result = []
    line = []
    code = []
    for token in tokenize(source, preserve_defines=True) + ["\n"]:
        if token != "\n":
            line.append(token)
            continue
        if line and line[0] in ("#", "%:"):
            if code:
                result.append((False, code))
                code = []
            result.append((True, line))
        else:
            code.extend(line)
        line = []
    if code:
        result.append((False, code))
    return result


def directive(tokens: list[str]) -> str:
    # chunks() encodes function-like macro names as 'F(' to preserve adjacency.
    if tokens[1:2] == ["define"]:
        value = join_tokens(tokens[3:])
        return "#define " + tokens[2] + (" " + value if value else "") + "\n"
    return join_tokens(tokens) + "\n"


def compact(source: str) -> str:
    return "".join(directive(tokens) if is_directive else join_tokens(tokens) + "\n"
                   for is_directive, tokens in chunks(source))


def select_paths(source: str, paths: tuple[str, ...]) -> str:
    """Filter simd.hpp's selection arms before expanding their dependencies.

    Keep the source's conditions and priority, not a second architecture table.
    An omitted scalar backend must never become an implicit fallback.
    """
    selection = re.compile(
        r'^#if defined\(__AVX512F__\).*?^#endif[^\n]*', re.MULTILINE | re.DOTALL)
    match = selection.search(source)
    if match is None:
        raise ValueError("cannot find SIMD backend selection in simd.hpp")
    arms = re.findall(r'^(#(?:if|elif|else)[^\n]*)\n(.*?)(?=^#(?:elif|else|endif)\b)',
                      match[0], re.MULTILINE | re.DOTALL)
    result = []
    found = set()
    conditional = False
    for condition, body in arms:
        backend = re.search(r'"fastpoly/detail/simd/(\w+)\.hpp"', body)
        if backend is None:
            raise ValueError("unexpected SIMD selection arm")
        path = backend[1]
        found.add(path)
        if path not in paths:
            continue
        if path == "scalar":
            if conditional:
                result.append("#else\n")
        else:
            result.append(re.sub(r'^#(?:if|elif)', '#elif' if conditional else '#if',
                                 condition) + "\n")
            conditional = True
        result.append(body)
    if found != set(PATHS):
        raise ValueError("unexpected SIMD backend set in simd.hpp")
    if conditional:
        if "scalar" not in paths:
            result.append('#else\n#error "fastpoly: no enabled SIMD path matches this target"\n')
        result.append("#endif")
    return source[:match.start()] + "".join(result) + source[match.end():]


def normalize_paths(paths=None) -> tuple[str, ...]:
    selected = set(PATHS if paths is None else paths)
    if not selected or selected - set(PATHS):
        raise ValueError("paths must be a nonempty subset of " + ", ".join(PATHS))
    return tuple(path for path in PATHS if path in selected)


def header_body(name: str, paths=None) -> str:
    """Inline local dependencies at their include sites, including backend arms.

    Unconditional dependencies are emitted once. Conditional includes get a
    branch-local set: including a header in one arm must not suppress it in
    another arm or later outside the conditional. Keep dependency guards too,
    so a conditional inclusion followed by an unconditional one stays valid.
    """
    paths = normalize_paths(paths)
    emitted = set()
    local_include = re.compile(r'\s*#\s*include\s+"fastpoly/([^"\n]+\.hpp)"\s*\n?')

    def expand(relative: str, seen: set[str], active: tuple[str, ...], *, outer=False):
        if relative in active:
            raise ValueError(f"cyclic local include: {' -> '.join((*active, relative))}")
        if relative in seen:
            return ""
        seen.add(relative)
        path = ROOT / "include" / "fastpoly" / relative
        guard = "FASTPOLY_" + relative[:-4].upper().replace("/", "_") + "_HPP"
        source = path.read_text(encoding="utf-8")
        if relative == "simd.hpp":
            source = select_paths(source, paths)
        lines = source.splitlines(keepends=True)
        opening = next((i for i, line in enumerate(lines) if line.strip() == f"#ifndef {guard}"), None)
        if opening is None or lines[opening + 1].strip() != f"#define {guard}":
            raise ValueError(f"unexpected include guard in {path}")
        last = max(i for i, line in enumerate(lines) if line.strip())
        if not re.fullmatch(r"#endif(?:\s*//.*)?", lines[last].strip()):
            raise ValueError(f"unexpected closing guard in {path}")
        result = [] if outer else lines[:opening + 2]
        depth = 0
        for line in lines[opening + 2:last]:
            if match := local_include.fullmatch(line):
                result.append(expand(match[1], seen if depth == 0 else seen.copy(),
                                     (*active, relative)))
            else:
                result.append(line)
                if re.match(r"\s*#\s*(?:if|ifdef|ifndef)\b", line):
                    depth += 1
                elif re.match(r"\s*#\s*endif\b", line):
                    depth -= 1
        if depth:
            raise ValueError(f"unbalanced conditional in {path}")
        if not outer:
            result.extend(lines[last:])
        return "".join(result)

    return expand(f"{name}.hpp", emitted, (), outer=True)


# The single-header distribution promises these spellings. Keep the complete
# poly/modint surfaces, including aliases, public constants and helper functions.
# Parameter names, template parameters and private members are not API names.
PUBLIC_NAMES = frozenset("""
    fpx poly Poly Mont ModInt mod998244353 mod1004535809 mod469762049
    mod167772161 mod754974721 mod1224736769 mod1000000007
    domain_error vec naive_threshold copy_prefix copy_range pointwise_mul
    trim add sub neg mul_scalar shift eval conv_limbs conv derivative integral
    inv_series inv log exp sqrt pow divmod div mod base value_type degree resize_
    mul to_ints from_ints inv_mod_2_32 primitive_root ninv one r2 inv2
    from_int raw reduce val raw_val is_zero str
""".split())
CPP_KEYWORDS = frozenset("""
    alignas alignof and and_eq asm auto bitand bitor bool break case catch char
    char8_t char16_t char32_t class compl concept const consteval constexpr
    constinit const_cast continue co_await co_return co_yield decltype default
    delete do double dynamic_cast else enum explicit export extern false float
    for friend goto if inline int long mutable namespace new noexcept not not_eq
    nullptr operator or or_eq private protected public register reinterpret_cast
    requires return short signed sizeof static static_assert static_cast struct
    switch template this thread_local throw true try typedef typeid typename
    union unsigned using virtual void volatile wchar_t while xor xor_eq final override
""".split())
# Names supplied by system headers without std:: qualification, and members of
# standard-library types. Shared spellings are kept everywhere: resolving a
# dependent member's type would require a full C++ compiler. New external APIs
# used by the maintained headers must be added here if not covered automatically.
EXTERNAL_NAMES = frozenset("""
    std uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t int64_t
    size_t ptrdiff_t nullptr_t NULL SIZE_MAX UINT32_MAX UINT64_MAX
    data size capacity empty begin end back front first second insert reserve
    resize clear pop_back push_back emplace_back get reset release swap
    max min value count digits
""".split())
EXTERNAL_SIMD = re.compile(
    r"(?:u?int|float|poly)\d+x\d+(?:x\d+)?_t|v\w+_(?:[sufp]\d+)(?:_[sufp]\d+)*")


def short_name(index: int) -> str:
    """C++ identifiers in length order, without reserved leading underscores."""
    first = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rest = first + "0123456789"
    width, count = 1, len(first)
    while index >= count:
        index -= count
        width += 1
        count *= len(rest)
    suffix = ""
    for _ in range(width - 1):
        suffix = rest[index % len(rest)] + suffix
        index //= len(rest)
    return first[index] + suffix


def implementation_renames(parts: list[tuple[bool, list[str]]]) -> dict[str, str]:
    """Rename implementation spellings across all headers, not just detail.

    Use one injective map for declarations and references, including dependent
    template members, constructor/destructor names and local variables. Preserve
    the public API manifest, C++/system names and all preprocessor identifiers.
    Names shared with an external/public API are conservatively kept globally.
    The pass is repository-specific; it is not a general C++ name resolver.
    """
    code = [token for is_directive, block in parts if not is_directive for token in block]
    protected = set(PUBLIC_NAMES | CPP_KEYWORDS | EXTERNAL_NAMES)
    protected.update(token.rstrip("(") for is_directive, block in parts if is_directive
                     for token in block)
    for i, token in enumerate(code):
        if (token.startswith("_") or EXTERNAL_SIMD.fullmatch(token)
                or (i >= 2 and code[i - 2:i] == ["std", "::"])):
            protected.add(token)
    counts = Counter(token for token in code if IDENTIFIER.fullmatch(token))
    used = set(counts) | protected
    names = sorted(counts.keys() - protected, key=lambda name: (-counts[name], name))
    mapping = {}
    index = 0
    for name in names:
        short = short_name(index)
        while short in used:
            index += 1
            short = short_name(index)
        if len(short) < len(name):
            mapping[name] = short
            used.add(short)
            index += 1
    return mapping


def rename_parts(parts, mapping):
    return [(is_directive, block if is_directive else [mapping.get(t, t) for t in block])
            for is_directive, block in parts]


def separate_includes(parts: list[tuple[bool, list[str]]]):
    """Hoist system includes, retaining their architecture conditionals.

    The complete branch prefix (if + preceding elif/else) selects the same arm
    as the source. Library guards are irrelevant to system-header selection.
    All system headers must be read before compression aliases are defined.
    """
    preamble, body, stack, seen = [], [], [], set()
    for is_directive, tokens in parts:
        if not is_directive:
            body.append((False, tokens))
            continue
        kind = tokens[1]
        if kind in ("if", "ifdef", "ifndef"):
            stack.append([tokens])
        elif kind in ("elif", "else"):
            stack[-1].append(tokens)
        elif kind == "endif":
            stack.pop()
        if kind == "include":
            if tokens[2:3] != ["<"]:
                raise ValueError(f"unexpanded local include: {join_tokens(tokens)}")
            context = [frame for frame in stack if not (
                frame[0][1] == "ifndef"
                and re.fullmatch(r"FASTPOLY_\w+_HPP", frame[0][2]))]
            key = (tuple(tuple(tuple(line) for line in frame) for frame in context), tuple(tokens))
            if key not in seen:
                seen.add(key)
                preamble.extend(directive(line) for frame in context for line in frame)
                preamble.append(directive(tokens))
                preamble.extend("#endif\n" for _ in context)
        else:
            body.append((True, tokens))
    if stack:
        raise ValueError("unbalanced conditional in expanded headers")
    return preamble, body


def macro_open(name: str, tokens: tuple[str, ...]) -> str:
    # Clang, GCC and MSVC support push/pop_macro. Restore caller definitions,
    # and keep compression aliases out of subsequent user/system headers.
    return (f'#pragma push_macro("{name}")\n#undef {name}\n'
            f'#define {name} {join_tokens(tokens)}\n')


def macro_close(name: str) -> str:
    return f'#undef {name}\n#pragma pop_macro("{name}")\n'


def macro_name(index: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    suffix = ""
    while True:
        suffix = alphabet[index % len(alphabet)] + suffix
        index //= len(alphabet)
        if not index:
            return "FP" + suffix


def replace(tokens: list[str], pattern: tuple[str, ...], name: str) -> list[str]:
    result = []
    i = 0
    n = len(pattern)
    while i < len(tokens):
        if tokens[i] == pattern[0] and tuple(tokens[i:i + n]) == pattern:
            result.append(name)
            i += n
        else:
            result.append(tokens[i])
            i += 1
    return result


def safe_pattern(pattern: tuple[str, ...]) -> bool:
    """Do not change argument collection for existing function-like macros.

    Argument collection precedes alias expansion. Thus aliases must not hide
    unmatched parentheses or outer commas. A leading '(' could also hide the
    invocation from the preceding macro name during the preprocessor's scan.
    """
    if pattern[0] == "(" or any("\n" in token for token in pattern):
        return False
    depth = 0
    for token in pattern:
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth < 0:
                return False
        elif token == "," and depth == 0:
            return False
    return depth == 0


def compress(code: list[list[str]]) -> tuple[list[list[str]], list[tuple[str, tuple[str, ...]]]]:
    """Greedily factor repeated token sequences, only accepting net byte savings.

    Definitions refer only to earlier aliases, forming an acyclic dictionary.
    Scoring includes all define/undef and macro save/restore bytes, plus exact
    whitespace at replacement boundaries. Directives never enter the dictionary.
    """
    dictionary = []
    used = {token for block in code for token in block}
    while True:
        index = len(dictionary)
        name = macro_name(index)
        while name in used:
            index += 1
            name = macro_name(index)
        counts = Counter()
        for block in code:
            for length in (*range(1, 17), 24, 32, 48, 64):
                counts.update(tuple(block[i:i + length])
                              for i in range(len(block) - length + 1))
        # A cheap upper bound prunes unique/short patterns before rendering.
        candidates = []
        for pattern, count in counts.items():
            if count < 2:
                continue
            length = sum(map(len, pattern))
            if count * (length + len(pattern) - 1 - len(name)) <= length + 100:
                continue
            if not safe_pattern(pattern):
                continue
            text = join_tokens(pattern)
            overhead = len(macro_open(name, pattern)) + len(macro_close(name))
            saving = count * (len(text) - len(name)) - overhead
            if saving > 0:
                candidates.append((saving, pattern, overhead))
        if not candidates:
            break
        candidates.sort(key=lambda item: (-item[0], item[1]))
        before = sum(len(join_tokens(block)) for block in code)
        best = None
        for _, pattern, overhead in candidates[:16]:
            replaced = [replace(block, pattern, name) for block in code]
            saving = before - sum(len(join_tokens(block)) for block in replaced) - overhead
            if saving > 0 and (best is None or saving > best[0]):
                best = (saving, pattern, replaced)
        if best is None:
            break
        _, pattern, code = best
        dictionary.append((name, pattern))
        used.add(name)
    return code, dictionary


def generate(paths=None) -> str:
    paths = normalize_paths(paths)
    parts = [part for name in HEADERS for part in chunks(header_body(name, paths))]
    parts = rename_parts(parts, implementation_renames(parts))
    preamble, body = separate_includes(parts)
    code, dictionary = compress([tokens for is_directive, tokens in body if not is_directive])
    it = iter(code)
    banner = (
        "// fastpoly - standalone C++20 header.\n"
        "// Generated by scripts/amalgamate.py; edit include/fastpoly/ headers.\n"
        f"// Paths: {', '.join(paths)}.\n"
        "#ifndef FASTPOLY_SINGLE_HPP\n#define FASTPOLY_SINGLE_HPP\n"
    )
    return (banner + "".join(preamble)
            + "".join(macro_open(name, pattern) for name, pattern in dictionary)
            + "".join(directive(tokens) if is_directive else join_tokens(next(it)) + "\n"
                      for is_directive, tokens in body)
            + "".join(macro_close(name) for name, _ in reversed(dictionary)) + "#endif\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "fastpoly.hpp")
    parser.add_argument("--check", action="store_true", help="fail if the output needs regeneration")
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=PATHS,
                        help="backends to include (default: all); scalar enables fallback")
    args = parser.parse_args()
    generated = generate(args.paths)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != generated:
            print(f"{args.output} is stale; run scripts/amalgamate.py", file=sys.stderr)
            return 1
    else:
        args.output.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
