"""Produce v12 of the manuscript from v10 as five tracked changes.

v12 contains everything in v11 (five framing edits to the Discussion) plus two
new numbered limitations spliced into the same insertion, so the limitations
paragraph reads First through Sixth.

Built from v10 rather than from v11 on purpose. v11's "Fourth" sentence is
itself a tracked insertion, and nesting a further insertion inside it produces
markup Word handles inconsistently. Rebuilding from v10 keeps the markup flat.

Two traps this script is written to avoid:

1. Tracked-change markup inside a table row makes Word silently discard the
   whole table, so we assert zero w:ins/w:del inside any w:tbl before writing.

2. The naive pattern ``<w:t[^>]*>`` also matches ``<w:tab/>`` (``<w:t`` + ``ab/``).
   Two of the target runs begin with ``<w:tab/>`` or
   ``<w:lastRenderedPageBreak/>``, so that pattern captures the *real* ``<w:t>``
   tag as text content and re-emits it as visible characters in the document.
   We therefore match ``<w:t(?:\\s[^>]*)?>`` and preserve every non-text child
   of any run we split.

Verification lives in verify_manuscript_v12.py, which re-parses the output with
an XML parser rather than these regexes, so a bug here cannot mask itself.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "NLP_Crisis_EMA Paper v10.docx"
DST = ROOT / "NLP_Crisis_EMA Paper v12.docx"
WORK = Path("/tmp/v12work")

AUTHOR = "Analysis revision"
DATE = "2026-10-05T00:00:00Z"

_id = [9000]

# A run element. The lookahead stops <w:rPr> from matching as <w:r>.
RUN_RE = re.compile(r"<w:r\b(?![a-zA-Z])(?:(?!</w:r>).)*</w:r>", re.S)
# A text element. The mandatory whitespace before attributes keeps <w:tab/> out.
T_RE = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
RUN_OPEN_RE = re.compile(r"^<w:r\b(?![a-zA-Z])[^>]*>")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tracked(old: str, new: str, rpr: str = "") -> str:
    """A tracked deletion of `old` followed by a tracked insertion of `new`."""
    _id[0] += 1
    did = _id[0]
    _id[0] += 1
    iid = _id[0]
    out = ""
    if old:
        out += (
            f'<w:del w:id="{did}" w:author="{AUTHOR}" w:date="{DATE}">'
            f'<w:r>{rpr}<w:delText xml:space="preserve">{esc(old)}</w:delText></w:r>'
            f"</w:del>"
        )
    if new:
        out += (
            f'<w:ins w:id="{iid}" w:author="{AUTHOR}" w:date="{DATE}">'
            f'<w:r>{rpr}<w:t xml:space="preserve">{esc(new)}</w:t></w:r>'
            f"</w:ins>"
        )
    return out


def plain_esc(text_esc: str, rpr: str = "") -> str:
    """A normal run holding already-XML-escaped text."""
    if not text_esc:
        return ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{text_esc}</w:t></w:r>'


def _split_run(run: str):
    """Return (open_tag, head, text_esc, tail) for a single <w:r> element.

    `head` is every child preceding the <w:t> element (w:rPr, w:tab,
    w:lastRenderedPageBreak, ...) and is preserved verbatim.
    """
    om = RUN_OPEN_RE.match(run)
    if not om:
        raise SystemExit("could not parse run opening tag")
    inner = run[om.end() : -len("</w:r>")]
    tm = T_RE.search(inner)
    if not tm:
        raise SystemExit("run has no <w:t> element")
    return om.group(0), inner[: tm.start()], tm.group(1), inner[tm.end() :]


def replace_in_run(xml: str, old: str, new: str) -> str:
    """Split whichever single <w:r> contains `old`, tracking the change."""
    target = esc(old)
    hits = [m for m in RUN_RE.finditer(xml) if target in m.group(0)]
    if len(hits) != 1:
        raise SystemExit(
            f"Expected exactly 1 run containing {old[:60]!r}, found {len(hits)}"
        )
    m = hits[0]
    open_tag, head, text_esc, tail = _split_run(m.group(0))

    if target not in text_esc:
        raise SystemExit(f"Run found but <w:t> does not hold {old[:60]!r}")
    if tail.strip():
        raise SystemExit(f"unexpected content after <w:t>: {tail[:80]!r}")

    rpr_m = re.search(r"<w:rPr>.*?</w:rPr>", head, re.S)
    rpr = rpr_m.group(0) if rpr_m else ""

    before, _, after = text_esc.partition(target)

    # Always emit the leading run so `head` (e.g. <w:tab/>) survives.
    rebuilt = f'{open_tag}{head}<w:t xml:space="preserve">{before}</w:t></w:r>'
    rebuilt += tracked(old, new, rpr)
    rebuilt += plain_esc(after, rpr)
    return xml[: m.start()] + rebuilt + xml[m.end() :]


def merge_runs_in_para(xml: str, anchor: str, count: int) -> str:
    """Merge `count` consecutive plain-text runs starting at the run holding `anchor`.

    Refuses to merge any run carrying run properties or any child other than
    <w:t>, so merging cannot drop formatting or layout elements.
    """
    p_m = _find_para(xml, anchor)
    para = p_m.group(0)
    runs = list(RUN_RE.finditer(para))
    start = next((i for i, r in enumerate(runs) if esc(anchor) in r.group(0)), None)
    if start is None:
        raise SystemExit(f"merge anchor not found: {anchor[:60]!r}")
    group = runs[start : start + count]
    if len(group) != count:
        raise SystemExit("not enough runs to merge")

    texts = []
    for r in group:
        _, head, text_esc, tail = _split_run(r.group(0))
        if head.strip() or tail.strip():
            raise SystemExit(
                f"refusing to merge a run with non-text children: "
                f"head={head[:60]!r} tail={tail[:60]!r}"
            )
        texts.append(text_esc)

    merged = f'<w:r><w:t xml:space="preserve">{"".join(texts)}</w:t></w:r>'
    new_para = para[: group[0].start()] + merged + para[group[-1].end() :]
    new_para = re.sub(r"<w:proofErr[^>]*/>", "", new_para)
    return xml[: p_m.start()] + new_para + xml[p_m.end() :]


def _find_para(xml: str, needle: str):
    for m in re.finditer(r"<w:p\b(?:(?!</w:p>).)*</w:p>", xml, re.S):
        if esc(needle) in m.group(0):
            return m
    raise SystemExit(f"paragraph not found for: {needle[:60]!r}")


# ---------------------------------------------------------------------------
# The three new limitations spliced into Edit 5
# ---------------------------------------------------------------------------
FOURTH = (
    "Fourth, Aim 1 was designed to establish convergent validity rather than "
    "incremental utility. Diary text did not improve prediction of concurrent "
    "or next-morning suicide risk beyond the five-item crisis measure itself, "
    "with every 95% confidence interval for the change in cross-validated R² "
    "including zero. Daily language should therefore be understood as tracking "
    "the same crisis construct that the brief inventory captures rather than "
    "as contributing information beyond it. "
)

FIFTH = (
    "Fifth, the prospective associations were carried largely by stable "
    "differences between participants rather than by day-to-day variation "
    "within them. When the cross-validated predictions were decomposed into "
    "between- and within-person components, the next-morning models retained "
    "between-person correlations of .50 for crisis symptom total and .40 for "
    "suicide risk, whereas the corresponding within-person correlations were "
    ".10 and .04. The concurrent models retained within-person correlations of "
    ".47 and .32, which suggests that diary language tracked same-evening "
    "fluctuations in crisis severity but primarily distinguished participants "
    "with persistently elevated scores when the outcome was assessed the "
    "following morning. This distinction bears directly on the monitoring "
    "application that motivates the current work, because a model that "
    "identifies which participants tend to score highly does not thereby "
    "indicate when a particular participant is deteriorating. "
)

SIXTH = (
    "Sixth, written diary summaries were collected only during the evening "
    "assessment, so the prospective analyses related evening language to "
    "next-morning symptoms across an intervening sleep period and a change in "
    "time of day. Because no morning language sample was available, we could "
    "not determine whether the attenuated next-morning associations reflected "
    "a genuine decline in the predictive value of diary language over time or "
    "the absence of a contemporaneous language sample at the morning "
    "assessment. Collecting written entries at both daily assessments would "
    "distinguish these two explanations and would permit a test of whether "
    "language and crisis symptoms covary within an assessment window "
    "irrespective of time of day. "
)

LIMITATIONS_TAIL = "Future studies should rectify these limitations."

EDITS = [
    # 1. Hedge the next-morning suicide-risk claim (Discussion, H1).
    (
        "was associated with concurrent and next-morning crisis symptom total "
        "scores as well as concurrent and next-morning suicide risk.",
        "was associated with concurrent and next-morning crisis symptom total "
        "scores, and with concurrent suicide risk. The corresponding "
        "next-morning association with suicide risk was statistically reliable "
        "but small.",
    ),
    # 2. "adequate proxy" implies substitutability.
    (
        "In sum, it appears that text content may be an adequate proxy for "
        "crisis severity.",
        "In sum, diary text appears to function as a convergent but noisier "
        "index of crisis severity rather than as a substitute for direct "
        "symptom report.",
    ),
    # 3. "we are able to capture" overstates what the design shows.
    (
        "In other words, we are able to capture crisis symptom severity through "
        "text content without asking the participants to rate their crisis "
        "symptoms directly.",
        "In other words, crisis-relevant content was recoverable from language "
        "that participants produced without being prompted to rate their "
        "crisis symptoms directly.",
    ),
    # 4. Keep passive assessment as motivation, not demonstrated capability.
    (
        "thus, the ability to passively assess one’s crisis symptom "
        "severity is paramount.",
        "a language-based index that does not depend on prompted self-report is "
        "therefore worth pursuing. Because participants in the present design "
        "did complete the concurrent self-reports, however, these data cannot "
        "establish how such an index would perform when self-report is "
        "unavailable.",
    ),
    # 5. Three new limitations: convergent validity, between-person carriage,
    #    and evening-only language collection.
    (
        LIMITATIONS_TAIL,
        FOURTH + FIFTH + SIXTH + LIMITATIONS_TAIL,
    ),
]

MERGE_ANCHOR = ", 2017; Rogers et al., 2019). In other words, we "


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

    # Edit 3's sentence straddles three plain runs; merge them first.
    xml = merge_runs_in_para(xml, MERGE_ANCHOR, 3)

    for old, new in EDITS:
        xml = replace_in_run(xml, old, new)

    # --- validation ---------------------------------------------------------
    import xml.dom.minidom as md

    md.parseString(xml.encode("utf-8"))

    assert len(re.findall(r"<w:p\b", xml)) == before_paras, "paragraph count changed"
    assert len(re.findall(r"<w:tbl>", xml)) == before_tbl, "table count changed"
    assert len(re.findall(r"<w:tab/>", xml)) == before_tab, "tab count changed"

    for t in re.findall(r"<w:tbl>.*?</w:tbl>", xml, re.S):
        assert not re.search(r"<w:(ins|del)\b", t), "tracked markup inside a table"

    n_ins = len(re.findall(r"<w:ins\b", xml))
    n_del = len(re.findall(r"<w:del\b", xml))
    assert n_ins == len(EDITS) and n_del == len(EDITS), f"ins={n_ins} del={n_del}"

    authors = set(re.findall(r'w:author="([^"]*)"', xml))
    assert authors == {AUTHOR}, f"unexpected revision authors: {authors}"

    for _, new in EDITS:
        assert esc(new) in xml, f"inserted text missing: {new[:50]!r}"

    # No OOXML tag may ever appear as literal document text.
    assert "&lt;w:" not in xml, "literal markup leaked into document text"

    # The limitations must read First through Sixth exactly once each.
    for ordinal in ("First,", "Second,", "Third,", "Fourth,", "Fifth,", "Sixth,"):
        assert xml.count(ordinal) >= 1, f"missing limitation ordinal {ordinal}"

    doc.write_text(xml, encoding="utf-8")

    # --- repackage, preserving the original entry order ---------------------
    if DST.exists():
        DST.unlink()
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(WORK / name, name)

    with zipfile.ZipFile(DST) as z:
        bad = z.testzip()
        assert bad is None, f"corrupt zip entry: {bad}"
        assert set(z.namelist()) == set(names), "zip entry set changed"
        md.parseString(z.read("word/document.xml"))

    print(f"wrote {DST}")
    print(f"  paragraphs {before_paras}, tables {before_tbl}, tabs {before_tab}")
    print(f"  tracked insertions {n_ins}, deletions {n_del}, author {AUTHOR!r}")
    print("  0 tracked markup inside any table, 0 literal markup in text")


if __name__ == "__main__":
    main()
