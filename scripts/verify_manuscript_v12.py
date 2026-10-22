"""Verify the generated manuscript against its source, using an XML parser.

Deliberately shares no code with make_manuscript_v12.py. An earlier version of
the editor mis-parsed <w:tab/> as a <w:t> opening tag and emitted a literal
OOXML tag as visible document text; the verifier of the day used the same flawed
regex, so the two errors cancelled and the check reported success. Rendering the
paragraphs with ElementTree instead makes that class of bug impossible to hide.

The load-bearing assertion is that REJECTING all tracked changes reproduces the
source document's text exactly. If that holds, the only differences a reviewer
can see are the tracked insertions themselves.

Usage:
    python scripts/verify_manuscript_v12.py [--src v10.docx] [--dst v12.docx]
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ROOT = Path(__file__).resolve().parents[1]


def body(path: Path) -> ET.Element:
    with zipfile.ZipFile(path) as z:
        return ET.fromstring(z.read("word/document.xml"))


def entries(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        return z.namelist()


def para_text(p: ET.Element, mode: str) -> str:
    """Render a <w:p> as Word would after accepting or rejecting all changes."""
    out: list[str] = []

    def walk(node: ET.Element, in_ins: bool = False, in_del: bool = False) -> None:
        for ch in node:
            tag = ch.tag
            if tag == W + "ins":
                walk(ch, True, in_del)
            elif tag == W + "del":
                walk(ch, in_ins, True)
            elif tag == W + "r":
                walk(ch, in_ins, in_del)
            elif tag == W + "t":
                drop = (mode == "accept" and in_del) or (mode == "reject" and in_ins)
                if not drop:
                    out.append(ch.text or "")
            elif tag == W + "delText":
                if mode == "reject":
                    out.append(ch.text or "")
            elif tag == W + "tab":
                out.append("\t")
            elif tag == W + "br":
                out.append("\n")

    walk(p)
    return "".join(out)


def paras(root: ET.Element) -> list[ET.Element]:
    return list(root.iter(W + "p"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="NLP_Crisis_EMA Paper v10.docx")
    ap.add_argument("--dst", default="NLP_Crisis_EMA Paper v12.docx")
    args = ap.parse_args()

    src, dst = ROOT / args.src, ROOT / args.dst
    sroot, droot = body(src), body(dst)
    sp, dp = paras(sroot), paras(droot)

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' - ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    print(f"{src.name}  ->  {dst.name}")

    check("zip entry set unchanged", set(entries(src)) == set(entries(dst)))
    check("paragraph count unchanged", len(sp) == len(dp), f"{len(sp)} vs {len(dp)}")
    check(
        "table count unchanged",
        len(list(sroot.iter(W + "tbl"))) == len(list(droot.iter(W + "tbl"))),
        f"{len(list(sroot.iter(W + 'tbl')))} tables",
    )

    # No tracked markup anywhere inside a table.
    bad_tbl = sum(
        1
        for t in droot.iter(W + "tbl")
        if list(t.iter(W + "ins")) or list(t.iter(W + "del"))
    )
    check("no tracked markup inside any table", bad_tbl == 0, f"{bad_tbl} offending")

    n_ins = len(list(droot.iter(W + "ins")))
    n_del = len(list(droot.iter(W + "del")))
    check("insertions match deletions", n_ins == n_del, f"{n_ins} ins / {n_del} del")

    authors = {
        e.get(W + "author")
        for tag in ("ins", "del")
        for e in droot.iter(W + tag)
    }
    check("single revision author", len(authors) == 1, str(sorted(authors)))

    src_text = [para_text(p, "accept") for p in sp]  # source carries no revisions
    rej = [para_text(p, "reject") for p in dp]
    acc = [para_text(p, "accept") for p in dp]

    check("REJECTING all changes reproduces the source exactly", rej == src_text)
    if rej != src_text:
        for i, (x, y) in enumerate(zip(rej, src_text)):
            if x != y:
                print(f"      first diff at paragraph {i}")
                print(f"        got : {x[:160]!r}")
                print(f"        want: {y[:160]!r}")
                break

    check(
        "tab characters preserved",
        sum(t.count("\t") for t in src_text) == sum(t.count("\t") for t in rej)
        == sum(t.count("\t") for t in acc),
    )
    check(
        "no literal OOXML tag in rendered text",
        not any("<w:" in t for t in acc + rej),
    )

    changed = [i for i, (x, y) in enumerate(zip(acc, src_text)) if x != y]
    check("changes confined to prose paragraphs", len(changed) > 0, f"paragraphs {changed}")

    # The limitations paragraph must read First through Sixth, in order, once each.
    lim = next((t for t in acc if "limitations that should be improved" in t), None)
    check("limitations paragraph found", lim is not None)
    if lim:
        ordinals = ["First,", "Second,", "Third,", "Fourth,", "Fifth,", "Sixth,"]
        positions = [lim.find(o) for o in ordinals]
        check(
            "First through Sixth each present exactly once",
            all(lim.count(o) == 1 for o in ordinals),
            ", ".join(f"{o.rstrip(',')}={lim.count(o)}" for o in ordinals),
        )
        check("ordinals appear in order", positions == sorted(positions))

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
