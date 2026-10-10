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


def header_body(name: str) -> str:
    """Inline local dependencies at their include sites, including backend arms.

    Unconditional dependencies are emitted once. Conditional includes get a
    branch-local set: including a header in one arm must not suppress it in
    another arm or later outside the conditional. Keep dependency guards too,
    so a conditional inclusion followed by an unconditional one stays valid.
    """
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
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
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


def generate() -> str:
    parts = [part for name in HEADERS for part in chunks(header_body(name))]
    preamble, body = separate_includes(parts)
    code, dictionary = compress([tokens for is_directive, tokens in body if not is_directive])
    it = iter(code)
    banner = (
        "// fastpoly - standalone C++20 header.\n"
        "// Generated by scripts/amalgamate.py; edit include/fastpoly/ headers.\n"
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
    args = parser.parse_args()
    generated = generate()
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != generated:
            print(f"{args.output} is stale; run scripts/amalgamate.py", file=sys.stderr)
            return 1
    else:
        args.output.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
