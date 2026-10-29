"""Check NLP_Crisis_EMA Paper v14.docx against v13, v12, and v10.

Shares no code with either editor. The accept/reject machinery is imported from
verify_manuscript_v13 so there is one implementation of those semantics, which
is the part most likely to hide a subtle bug, and that module parses with
ElementTree rather than with the editors' regexes.

The checks that carry the weight:

  - Rejecting the "Copyedit revision" layer must still reproduce v12 exactly,
    and rejecting both layers must reproduce v10 exactly. v14 adds to the same
    author layer as v13, so neither baseline may move.
  - Each inserted DOI must sit inside a <w:hyperlink> whose relationship Target
    equals the text the reader sees. A hyperlink whose label and target disagree
    is the one failure here that proofreading would not catch.
  - Every DOI in the reference list is checked the same way, not just the two
    that were added.
"""

from __future__ import annotations

import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from verify_manuscript_v13 import (
    V12_AUTHOR,
    V13_AUTHOR,
    accept_all,
    load,
    para_texts,
    q,
    reject,
)

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "NLP_Crisis_EMA Paper v10.docx"
V12 = ROOT / "NLP_Crisis_EMA Paper v12.docx"
V13 = ROOT / "NLP_Crisis_EMA Paper v13.docx"
V14 = ROOT / "NLP_Crisis_EMA Paper v14.docx"

R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

ADDED = {
    "https://doi.org/10.1111/sltb.12439",
    "https://doi.org/10.4088/JCP.24m15320",
}

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def relationships(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/_rels/document.xml.rels"))
    return {el.get("Id"): el.get("Target") for el in root}


def hyperlink_pairs(root: ET.Element) -> list[tuple[str, str]]:
    """(relationship id, displayed text) for every hyperlink in the document."""
    out = []
    for h in root.iter(q("hyperlink")):
        rid = h.get(f"{{{R}}}id")
        if rid:
            out.append((rid, "".join(t.text or "" for t in h.iter(q("t")))))
    return out


def raw(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        return z.read("word/document.xml").decode("utf-8")


def main() -> None:
    print(f"{V13.name}  ->  {V14.name}")

    for p in (V10, V12, V13, V14):
        if not p.exists():
            check(False, f"{p.name} present")
            print("missing input")
            sys.exit(1)

    with zipfile.ZipFile(V13) as a, zipfile.ZipFile(V14) as b:
        check(set(a.namelist()) == set(b.namelist()), "zip entry set unchanged")

    src, dst = load(V13), load(V14)
    src_paras, dst_paras = para_texts(src), para_texts(dst)
    check(len(src_paras) == len(dst_paras), "paragraph count unchanged",
          f"{len(dst_paras)}")
    check(len(list(src.iter(q("tbl")))) == len(list(dst.iter(q("tbl")))),
          "table count unchanged", f"{len(list(dst.iter(q('tbl'))))} tables")

    offending = sum(
        len(list(tbl.iter(q(tag)))) for tbl in dst.iter(q("tbl")) for tag in ("ins", "del")
    )
    check(offending == 0, "no tracked markup inside any table", f"{offending}")

    # --- revision bookkeeping ----------------------------------------------
    counts: dict[tuple[str, str], int] = {}
    for tag in ("ins", "del"):
        for el in dst.iter(q(tag)):
            key = (el.get(q("author")) or "", tag)
            counts[key] = counts.get(key, 0) + 1
    check(
        {a for a, _ in counts} == {V12_AUTHOR, V13_AUTHOR},
        "two revision authors",
        str(sorted({a for a, _ in counts})),
    )
    check(
        counts.get((V12_AUTHOR, "ins")) == 5 and counts.get((V12_AUTHOR, "del")) == 5,
        "v12 layer unchanged",
        f"{counts.get((V12_AUTHOR, 'ins'))} ins / {counts.get((V12_AUTHOR, 'del'))} del",
    )
    check(
        counts.get((V13_AUTHOR, "ins")) == 12 and counts.get((V13_AUTHOR, "del")) == 10,
        "copyedit layer grew by 5 ins / 1 del",
        f"{counts.get((V13_AUTHOR, 'ins'))} ins / {counts.get((V13_AUTHOR, 'del'))} del",
    )

    # --- the baselines must not have moved ---------------------------------
    v12_paras = para_texts(load(V12))
    rejected = para_texts(reject(dst, {V13_AUTHOR}))
    diff = [i for i, (a, b) in enumerate(zip(v12_paras, rejected)) if a != b]
    check(rejected == v12_paras, "REJECTING the copyedit layer reproduces v12 exactly",
          f"differing: {diff[:6]}" if diff else "")

    v10_paras = para_texts(load(V10))
    all_rej = para_texts(reject(dst, None))
    diff10 = [i for i, (a, b) in enumerate(zip(v10_paras, all_rej)) if a != b]
    check(all_rej == v10_paras, "REJECTING both layers reproduces v10 exactly",
          f"differing: {diff10[:6]}" if diff10 else "")

    accepted = para_texts(accept_all(dst))
    body = re.sub(r"\s+", " ", " ".join(accepted))

    # --- the DOIs -----------------------------------------------------------
    for url in sorted(ADDED):
        check(url in body, f"accepted text contains {url}")
    check("413–422." in body, "2019 page range uses an en dash")
    check("413-422." not in body, "2019 hyphen page range is gone")
    check("The Narrative‐Crisis Model of suicide" not in body,
          "Cohen (2022) still absent from accepted text")

    rels = relationships(V14)
    pairs = hyperlink_pairs(dst)
    mismatched = [
        (rid, text, rels.get(rid))
        for rid, text in pairs
        if text.startswith("http") and rels.get(rid) != text
    ]
    check(not mismatched, "every hyperlink label matches its relationship target",
          f"{len(pairs)} links checked" if not mismatched else str(mismatched[:3]))

    linked = {text for _, text in pairs}
    check(ADDED <= linked, "both new DOIs are hyperlinks, not plain text",
          f"missing: {sorted(ADDED - linked)}" if not ADDED <= linked else "")

    # DOI suffixes legitimately contain parentheses, so only whitespace ends
    # one. A trailing sentence period is not part of the DOI.
    def plain_text_dois(doc: ET.Element) -> set[str]:
        text = re.sub(r"\s+", " ", " ".join(para_texts(accept_all(doc))))
        found = {t.rstrip(".") for t in re.findall(r"https://doi\.org/\S+", text)}
        live = {t.rstrip(".") for _, t in hyperlink_pairs(doc)}
        return found - live

    before = plain_text_dois(src)
    after = plain_text_dois(dst)
    check(after == before, "no DOI was added as plain text",
          f"{len(after)} pre-existing: {sorted(after)}"
          if after == before else f"new: {sorted(after - before)}")

    # --- markup legality ----------------------------------------------------
    x = raw(V14)
    nested = [
        m.group(1)[:60]
        for m in re.finditer(r"<w:ins\b[^>]*>(.*?)</w:ins>", x, re.S)
        if "<w:ins" in m.group(1)
    ]
    check(not nested, "no nested tracked insertion")
    bad_link = [
        tag
        for tag in ("ins", "del")
        for m in re.finditer(rf"<w:{tag}\b[^>]*>(.*?)</w:{tag}>", x, re.S)
        if "<w:hyperlink" in m.group(1)
    ]
    check(not bad_link, "no hyperlink inside tracked markup")

    orphan = [rid for rid, _ in pairs if rid not in rels]
    check(not orphan, "no hyperlink points at a missing relationship", str(orphan[:3]))

    # --- residual hygiene ---------------------------------------------------
    src_tabs = sum(len(list(p.iter(q("tab")))) for p in src.iter(q("p")))
    dst_tabs = sum(len(list(p.iter(q("tab")))) for p in dst.iter(q("p")))
    check(src_tabs == dst_tabs, "tab elements preserved", f"{dst_tabs} tabs")
    check("<w:" not in body, "no literal OOXML tag in rendered text")

    changed = sorted(
        {
            i
            for i, p in enumerate(dst.iter(q("p")))
            if any(list(p.iter(q(tag))) for tag in ("ins", "del"))
        }
    )
    check(changed == [20, 59, 88, 92, 96, 109, 110, 111, 659],
          "changes confined to the expected paragraphs", str(changed))

    print("all checks passed" if not failures else f"{len(failures)} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
