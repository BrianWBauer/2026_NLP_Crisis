"""Produce v16 from v14: six copyediting errors and three dash problems.

All nine were found by reading the rendered PDF, which is the only way any of
them could have been found. Every one is invisible to the structural checks the
other verifiers run.

The six errors:

  - "have risen in prominence to (e.g., Kleiman et al., 2018)", a sentence left
    with a dangling preposition.
  - "negative emotional valence, used VADER", missing its subject.
  - "VADER sentiment along was used", for "alone".
  - "post-hot selection", for "post-hoc".
  - "Supplementary Table S1" cited for the verbal artifact control, which is
    Table S2. Table S1 is the OLS sensitivity analysis.
  - "Long 1997", missing its comma.

The three dash problems are hyphens standing in for parenthetical dashes, the
same defect already repaired in the Methods in v13. Two remain in the Aim 3
summary and in the first limitation, and one page range reads "PC1 - PC5" where
the rest of the manuscript uses an en dash.

Neither dash span can take a pair of commas without ambiguity. The Aim 3 span
encloses a comma-separated list, so it becomes parentheses; the limitation span
already contains a parenthetical citation, so it becomes commas.

The limitation span straddles five runs, two of which hold nothing but a hyphen,
so those are merged before the replacement. That paragraph also carries v12's
tracked insertion further along, and the merge touches only the five leading
runs, well clear of it.

These join the existing "Copyedit revision" layer rather than opening a third,
so rejecting that layer still lands exactly on v12.

Verification lives in verify_manuscript_v16.py.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import make_manuscript_v13 as v13

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "NLP_Crisis_EMA Paper v14.docx"
DST = ROOT / "NLP_Crisis_EMA Paper v16.docx"
WORK = Path("/tmp/v16work")

AUTHOR = "Copyedit revision"
DATE = "2026-10-29T00:00:00Z"

v13.AUTHOR = AUTHOR
v13.DATE = DATE
v13._id[0] = 23000

esc = v13.esc

# The limitation's dash span sits across five runs, two of which are bare
# hyphens. Merge them so the whole sentence lives in one <w:t>.
#
# The merge has to start at the paragraph's first run, because the space before
# the opening hyphen belongs to that run and the replacement has to consume it;
# anchoring one run later would leave " , detecting". That first run carries the
# paragraph's indent <w:tab/>, which v13's merge helper refuses outright and
# which must not be dropped: it is visible layout and it is counted by every
# verifier. So the merge below preserves it at the head of the merged run, where
# it was already, rather than discarding it.
MERGES = [("whereas the clinical motivation for this work", 0, 5)]

# Empty elements that may be carried through a merge, kept in document order at
# the head of the merged run. Anything else aborts the merge.
CARRYABLE = ("<w:tab/>", "<w:lastRenderedPageBreak/>")


def merge_run_range(xml: str, para_anchor: str, start: int, count: int) -> str:
    """Merge `count` runs beginning at index `start` within one paragraph.

    Unlike the v13 helper this selects runs by position rather than by a text
    anchor, which is what lets it merge runs holding nothing but a hyphen.
    """
    p_m = v13._find_para(xml, para_anchor)
    para = p_m.group(0)
    runs = list(v13.RUN_RE.finditer(para))
    group = runs[start : start + count]
    if len(group) != count:
        raise SystemExit(f"paragraph has {len(runs)} runs; cannot merge {count} at {start}")

    texts, carried = [], []
    for r in group:
        _, head, text_esc, tail = v13._split_run(r.group(0))
        rest = head
        for el in CARRYABLE:
            if el in rest:
                carried.append(el)
                rest = rest.replace(el, "")
        if rest.strip() or tail.strip():
            raise SystemExit(
                f"refusing to merge a run with non-text children: "
                f"head={rest[:60]!r} tail={tail[:60]!r}"
            )
        texts.append(text_esc)

    merged = (
        f'<w:r>{"".join(carried)}<w:t xml:space="preserve">{"".join(texts)}</w:t></w:r>'
    )
    new_para = para[: group[0].start()] + merged + para[group[-1].end() :]
    new_para = re.sub(r"<w:proofErr[^>]*/>", "", new_para)
    return xml[: p_m.start()] + new_para + xml[p_m.end() :]

EDITS = [
    # --- the six errors ---------------------------------------------------
    (
        "dangling preposition",
        "risen in prominence to (e.g.,",
        "risen in prominence (e.g.,",
    ),
    (
        "missing subject",
        "negative emotional valence, used VADER",
        "negative emotional valence, we used VADER",
    ),
    (
        "along/alone",
        " sentiment along was used",
        " sentiment alone was used",
    ),
    (
        "post-hot/post-hoc",
        "post-hot selection",
        "post-hoc selection",
    ),
    (
        "wrong supplementary table",
        "; Supplementary Table S1)",
        "; Supplementary Table S2)",
    ),
    (
        "missing comma in citation",
        "; Long 1997)",
        "; Long, 1997)",
    ),
    # --- the three dash problems ------------------------------------------
    (
        "Aim 3 parenthetical dashes",
        "crisis-relevant domains - entrapment and isolation, hopelessness and "
        "affective collapse, and cognitive constriction and ruminative "
        "problem-solving failure - suggesting",
        "crisis-relevant domains (entrapment and isolation, hopelessness and "
        "affective collapse, and cognitive constriction and ruminative "
        "problem-solving failure), suggesting",
    ),
    (
        "limitation parenthetical dashes",
        "whereas the clinical motivation for this work - detecting crisis in "
        "individuals who may not disclose ideation prior to a suicide attempt "
        "or death (Cohen et al., 2024) - concerns higher-acuity populations",
        "whereas the clinical motivation for this work, detecting crisis in "
        "individuals who may not disclose ideation prior to a suicide attempt "
        "or death (Cohen et al., 2024), concerns higher-acuity populations",
    ),
    (
        "PC range en dash",
        "for PC1 - PC5,",
        "for PC1–PC5,",
    ),
]


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(SRC) as z:
        names = z.namelist()
        z.extractall(WORK)

    doc = WORK / "word" / "document.xml"
    xml = doc.read_text(encoding="utf-8")

    before_paras = len(re.findall(r"<w:p\b", xml))
    before_tbl = len(re.findall(r"<w:tbl>", xml))
    before_tab = len(re.findall(r"<w:tab/>", xml))
    before_ins = len(re.findall(r"<w:ins\b", xml))
    before_del = len(re.findall(r"<w:del\b", xml))

    for anchor, start, count in MERGES:
        xml = merge_run_range(xml, anchor, start, count)

    for label, old, new in EDITS:
        try:
            xml = v13.replace_in_run(xml, old, new)
        except SystemExit as exc:
            raise SystemExit(f"[{label}] {exc}") from exc

    # --- validation ---------------------------------------------------------
    import xml.dom.minidom as md

    md.parseString(xml.encode("utf-8"))

    assert len(re.findall(r"<w:p\b", xml)) == before_paras, "paragraph count changed"
    assert len(re.findall(r"<w:tbl>", xml)) == before_tbl, "table count changed"
    assert len(re.findall(r"<w:tab/>", xml)) == before_tab, "tab count changed"

    for t in re.findall(r"<w:tbl>.*?</w:tbl>", xml, re.S):
        assert not re.search(r"<w:(ins|del)\b", t), "tracked markup inside a table"

    new_ins = len(re.findall(r"<w:ins\b", xml)) - before_ins
    new_del = len(re.findall(r"<w:del\b", xml)) - before_del
    assert new_ins == len(EDITS), f"new insertions = {new_ins}"
    assert new_del == len(EDITS), f"new deletions = {new_del}"

    for m in re.finditer(r"<w:ins\b[^>]*(?<!/)>.*?</w:ins>", xml, re.S):
        body = m.group(0)
        assert not re.search(r"<w:ins\b", body[body.index(">") + 1 : -len("</w:ins>")]), (
            "nested tracked insertion"
        )

    authors = set(re.findall(r'w:author="([^"]*)"', xml))
    assert authors == {"Analysis revision", AUTHOR}, f"authors: {authors}"

    for label, _, new in EDITS:
        assert esc(new) in xml, f"[{label}] inserted text missing"

    assert "&lt;w:" not in xml, "literal markup leaked into document text"

    doc.write_text(xml, encoding="utf-8")

    if DST.exists():
        DST.unlink()
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(WORK / name, name)

    with zipfile.ZipFile(DST) as z:
        assert z.testzip() is None, "corrupt zip entry"
        assert set(z.namelist()) == set(names), "zip entry set changed"
        md.parseString(z.read("word/document.xml"))

    print(f"wrote {DST}")
    print(f"  paragraphs {before_paras}, tables {before_tbl}, tabs {before_tab}")
    print(f"  new revisions: {new_ins} ins / {new_del} del, author {AUTHOR!r}")
    for label, _, _ in EDITS:
        print(f"    - {label}")


if __name__ == "__main__":
    main()
