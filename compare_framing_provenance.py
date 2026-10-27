#!/usr/bin/env python3
"""
compare_framing_provenance.py -- how close are our framings to their sources?

The paper is about to say the framings are our own implementations of
documented techniques rather than reproduced prompt text. That sentence should
be checked before it is printed. This measures the overlap and reports SCORES
ONLY: no prompt text from either side is printed or written.

    git clone https://github.com/elder-plinius/L1B3RT4S /tmp/l1b
    python compare_framing_provenance.py --ours attacks/l1b3rt4s_attacks.py \
                                         --theirs /tmp/l1b

What the numbers mean, on the normalized text (lowercased, whitespace and
punctuation collapsed):

  token_jaccard   shared vocabulary. High alone means the same subject matter.
  seq_ratio       difflib similarity of the whole string, 0 to 1.
  longest_common  the longest shared run of words. This is the one that
                  matters: a long shared run is copied text, whatever the
                  surrounding edits.

Rough reading: longest_common under about 8 words is ordinary technique
overlap; 15 or more is a quoted passage and should be attributed as such.
Judge each case yourself; this ranks candidates, it does not decide.
"""
from __future__ import annotations
import argparse, ast, difflib, re, sys
from pathlib import Path

WORD = re.compile(r"[a-z0-9]+")


def _flatten(node) -> str | None:
    """Text of a literal, an f-string, or literals joined by +."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values
                       if isinstance(v, ast.Constant) and isinstance(v.value, str))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l, r = _flatten(node.left), _flatten(node.right)
        return (l or "") + (r or "") if (l or r) else None
    return None


def norm(s: str) -> list[str]:
    return WORD.findall(s.lower())


def longest_common_run(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def strings_from_markdown(p: Path) -> dict[str, str]:
    """Framings from PROMPTS.md: ### `name` followed by a fenced block."""
    out, text = {}, p.read_text(errors="replace")
    for m in re.finditer(r"^###\s+`?([^`\n]+)`?\s*$", text, re.M):
        name = m.group(1).strip()
        rest = text[m.end():]
        nxt = re.search(r"^#{1,3}\s", rest, re.M)
        body = rest[: nxt.start()] if nxt else rest
        body = re.sub(r"^```[a-z]*\n?|```", "", body, flags=re.M)
        if len(body.strip()) > 40:
            out[name] = body.strip()
    return out


def strings_from_python(p: Path) -> dict[str, str]:
    """Every string literal of reasonable length, keyed by its assignment name."""
    out, tree = {}, ast.parse(p.read_text(errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            val = node.value
            flat = _flatten(val)
            if flat and len(flat) > 60:
                val = ast.Constant(value=flat)
            if isinstance(val, ast.Constant) and isinstance(val.value, str) and len(val.value) > 60:
                tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
                name = getattr(tgt, "id", None) or getattr(tgt, "attr", None)
                if name:
                    out[name] = val.value
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and isinstance(k.value, str)
                        and isinstance(v, ast.Constant) and isinstance(v.value, str)
                        and len(v.value) > 60):
                    out.setdefault(k.value, v.value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="python file holding our prompt strings")
    ap.add_argument("--theirs", required=True, help="directory of the source corpus")
    ap.add_argument("--top", type=int, default=3, help="closest matches to report per framing")
    a = ap.parse_args()

    op = Path(a.ours)
    ours = (strings_from_markdown(op) if op.suffix.lower() in {".md", ".markdown"}
            else strings_from_python(op))
    if not ours:
        return print(f"no prompt strings found in {a.ours}", file=sys.stderr) or 1
    print(f"extracted from {op.name}:")
    for n, t in sorted(ours.items()):
        print(f"   {n[:38]:38} {len(norm(t)):5d} tokens"
              f"{'   SKIPPED, under 20' if len(norm(t)) < 20 else ''}")
    print()

    theirs: list[tuple[str, list[str]]] = []
    for f in sorted(Path(a.theirs).rglob("*")):
        if f.is_file() and f.suffix.lower() in {".md", ".txt", ".mkd", ""} and f.stat().st_size < 2_000_000:
            try:
                for i, block in enumerate(re.split(r"\n\s*\n", f.read_text(errors="replace"))):
                    toks = norm(block)
                    if len(toks) >= 20:
                        theirs.append((f"{f.relative_to(a.theirs)}#{i}", toks))
            except Exception:
                continue
    print(f"ours: {len(ours)} prompt strings    theirs: {len(theirs)} blocks\n")
    if not theirs:
        return print("no corpus blocks found; check --theirs", file=sys.stderr) or 1

    print(f"{'our framing':30} {'jaccard':>8} {'seq':>6} {'run':>5}  closest source block")
    print("-" * 92)
    flagged, compared = [], 0
    for name, text in sorted(ours.items()):
        at = norm(text)
        if len(at) < 20:
            continue
        aset = set(at)
        scored = []
        for src, bt in theirs:
            inter = len(aset & set(bt))
            if not inter:
                continue
            jac = inter / len(aset | set(bt))
            scored.append((jac, src, bt))
        scored.sort(reverse=True, key=lambda r: r[0])
        for jac, src, bt in scored[:a.top]:
            seq = difflib.SequenceMatcher(None, at, bt).ratio()
            run = longest_common_run(at, bt)
            compared += 1
            print(f"{name[:30]:30} {jac:8.3f} {seq:6.3f} {run:5d}  {src}")
            if run >= 15:
                flagged.append((name, src, run))
        print()

    if flagged:
        print("Shared runs of 15 or more words. Read these yourself before claiming")
        print("the text was not reproduced:")
        for n, s, r in flagged:
            print(f"  {n}  <->  {s}   {r} words")
    elif compared == 0:
        print("NOTHING WAS COMPARED. Every source string was skipped or had no")
        print("vocabulary in common with the corpus, so this run is not evidence")
        print("either way. Point --ours at PROMPTS.md and re-run.")
        return 1
    else:
        print(f"No shared run of 15 or more words, over {compared} comparisons.")
        print("Consistent with reimplementation of the techniques rather than")
        print("reproduction of their text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())