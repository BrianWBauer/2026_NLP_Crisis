"""Produce v13 of the manuscript from v12 as a second layer of tracked changes.

v13 adds seven changes, none of them touching a paragraph that v12 already
edited, so the two revision layers stay flat and independent:

  1. Abstract. The occurrence-versus-severity sentence is restated so that it
     names the hurdle analysis and reports what the decomposition actually
     showed for a single between-person component.
  2. Methods. The model-comparison sentence loses the in-text AIC deltas and
     expected-zero counts, which Table S6 already reports in full.
  3. Methods. Two hyphens used as parenthetical dashes become commas.
  4. Table S6 note. Gains the AIC deltas removed from the Methods text.
  5. References. Cohen et al. (2022) is deleted; it is cited nowhere in the
     text.
  6. References. Cohen et al. (2024) gets its journal title cased correctly
     and its duplicated page range removed.

New changes carry a separate revision author so Word colours them apart from
the v12 Discussion edits, which Brian has already approved.

The two traps this file is written against, carried over from
make_manuscript_v12.py:

1. Tracked markup inside a table row makes Word silently discard the whole
   table, so we assert zero w:ins/w:del inside any w:tbl before writing. Every
   v13 target paragraph was checked to sit outside any w:tbl, including the
   Table S6 note, which is a body paragraph beneath the table rather than a
   cell.

2. The naive pattern ``<w:t[^>]*>`` also matches ``<w:tab/>`` (``<w:t`` +
   ``ab/``), which re-emits the real ``<w:t>`` tag as visible characters. We
   match ``<w:t(?:\\s[^>]*)?>`` and preserve every non-text child of any run we
   split.

A third trap is specific to v13: ``<w:hyperlink>`` belongs to EG_PContent and
is not valid inside ``<w:del>``, so deleting the Cohen (2022) entry wraps the
plain runs in one ``<w:del>`` and the hyperlink's own run in a second
``<w:del>`` nested inside the hyperlink.

Verification lives in verify_manuscript_v13.py, which re-parses the output with
an XML parser rather than these regexes, so a bug here cannot mask itself.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "NLP_Crisis_EMA Paper v12.docx"
DST = ROOT / "NLP_Crisis_EMA Paper v13.docx"
WORK = Path("/tmp/v13work")

AUTHOR = "Copyedit revision"
DATE = "2026-10-29T00:00:00Z"
V12_AUTHOR = "Analysis revision"

_id = [20000]

# A run element. The lookahead stops <w:rPr> from matching as <w:r>.
RUN_RE = re.compile(r"<w:r\b(?![a-zA-Z])(?:(?!</w:r>).)*</w:r>", re.S)
# A text element. The mandatory whitespace before attributes keeps <w:tab/> out.
T_RE = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)
T_OPEN_RE = re.compile(r"<w:t(?:\s[^>]*)?>")
RUN_OPEN_RE = re.compile(r"^<w:r\b(?![a-zA-Z])[^>]*>")
PARA_RE = re.compile(r"<w:p\b(?:(?!</w:p>).)*</w:p>", re.S)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _next_id() -> int:
    _id[0] += 1
    return _id[0]


def tracked(old: str, new: str, rpr: str = "") -> str:
    """A tracked deletion of `old` followed by a tracked insertion of `new`."""
    out = ""
    if old:
        out += (
            f'<w:del w:id="{_next_id()}" w:author="{AUTHOR}" w:date="{DATE}">'
            f'<w:r>{rpr}<w:delText xml:space="preserve">{esc(old)}</w:delText></w:r>'
            f"</w:del>"
        )
    if new:
        out += (
            f'<w:ins w:id="{_next_id()}" w:author="{AUTHOR}" w:date="{DATE}">'
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


def _find_para(xml: str, needle: str):
    """The single paragraph containing `needle`. Ambiguity is an error."""
    hits = [m for m in PARA_RE.finditer(xml) if esc(needle) in m.group(0)]
    if len(hits) != 1:
        raise SystemExit(
            f"Expected exactly 1 paragraph containing {needle[:60]!r}, "
            f"found {len(hits)}"
        )
    return hits[0]


def _one_run(xml: str, needle: str):
    target = esc(needle)
    hits = [m for m in RUN_RE.finditer(xml) if target in m.group(0)]
    if len(hits) != 1:
        raise SystemExit(
            f"Expected exactly 1 run containing {needle[:60]!r}, found {len(hits)}"
        )
    return hits[0]


def replace_in_run(xml: str, old: str, new: str) -> str:
    """Split whichever single <w:r> contains `old`, tracking the change."""
    target = esc(old)
    m = _one_run(xml, old)
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


def insert_after_in_run(xml: str, anchor: str, new: str) -> str:
    """Track a pure insertion immediately after `anchor` inside its own run."""
    target = esc(anchor)
    m = _one_run(xml, anchor)
    open_tag, head, text_esc, tail = _split_run(m.group(0))

    if target not in text_esc:
        raise SystemExit(f"Run found but <w:t> does not hold {anchor[:60]!r}")
    if tail.strip():
        raise SystemExit(f"unexpected content after <w:t>: {tail[:80]!r}")

    rpr_m = re.search(r"<w:rPr>.*?</w:rPr>", head, re.S)
    rpr = rpr_m.group(0) if rpr_m else ""

    before, _, after = text_esc.partition(target)

    rebuilt = f'{open_tag}{head}<w:t xml:space="preserve">{before}{target}</w:t></w:r>'
    rebuilt += tracked("", new, rpr)
    rebuilt += plain_esc(after, rpr)
    return xml[: m.start()] + rebuilt + xml[m.end() :]


def merge_runs_in_para(xml: str, anchor: str, count: int) -> str:
    """Merge `count` consecutive plain-text runs starting at the run holding `anchor`.

    Refuses to merge any run carrying run properties or any child other than
    <w:t>, so merging cannot drop formatting or layout elements. The single
    exception is <w:lastRenderedPageBreak/>, a cached pagination hint that Word
    recomputes on open; it is kept, but moves to the head of the merged run.
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

    texts, keep = [], ""
    for r in group:
        _, head, text_esc, tail = _split_run(r.group(0))
        head_rest = head.replace("<w:lastRenderedPageBreak/>", "")
        if "<w:lastRenderedPageBreak/>" in head:
            keep = "<w:lastRenderedPageBreak/>"
        if head_rest.strip() or tail.strip():
            raise SystemExit(
                f"refusing to merge a run with non-text children: "
                f"head={head_rest[:60]!r} tail={tail[:60]!r}"
            )
        texts.append(text_esc)

    merged = f'<w:r>{keep}<w:t xml:space="preserve">{"".join(texts)}</w:t></w:r>'
    new_para = para[: group[0].start()] + merged + para[group[-1].end() :]
    new_para = re.sub(r"<w:proofErr[^>]*/>", "", new_para)
    return xml[: p_m.start()] + new_para + xml[p_m.end() :]


def delete_paragraph(xml: str, anchor: str) -> str:
    """Mark an entire paragraph, including its paragraph mark, as deleted.

    Runs outside a hyperlink are wrapped in one <w:del>; a hyperlink's run gets
    its own <w:del> nested inside the hyperlink, because <w:hyperlink> is not
    valid content for <w:del>.
    """
    p_m = _find_para(xml, anchor)
    para = p_m.group(0)

    if re.search(r"<w:(ins|del)\b", para):
        raise SystemExit("refusing to delete a paragraph that already has revisions")

    para = re.sub(r"<w:proofErr[^>]*/>", "", para)

    def to_del_text(run: str) -> str:
        """Rewrite every <w:t> in a run as <w:delText>, keeping attributes."""
        out, pos = [], 0
        for tm in T_RE.finditer(run):
            open_m = T_OPEN_RE.match(run, tm.start())
            attrs = open_m.group(0)[len("<w:t") : -1]
            out.append(run[pos : tm.start()])
            out.append(f"<w:delText{attrs}>{tm.group(1)}</w:delText>")
            pos = tm.end()
        out.append(run[pos:])
        return "".join(out)

    def wrap(body: str) -> str:
        return (
            f'<w:del w:id="{_next_id()}" w:author="{AUTHOR}" w:date="{DATE}">'
            f"{body}</w:del>"
        )

    head_m = re.match(r"<w:p\b[^>]*>(?:<w:pPr>.*?</w:pPr>)?", para, re.S)
    if not head_m:
        raise SystemExit("could not parse paragraph opening")
    head, body = head_m.group(0), para[head_m.end() : -len("</w:p>")]

    # Walk the body, batching consecutive runs and handling hyperlinks apart.
    out, pos, batch = [], 0, []

    def flush() -> None:
        if batch:
            out.append(wrap("".join(batch)))
            batch.clear()

    for m in re.finditer(
        r"<w:hyperlink\b(?:(?!</w:hyperlink>).)*</w:hyperlink>"
        r"|<w:r\b(?![a-zA-Z])(?:(?!</w:r>).)*</w:r>",
        body,
        re.S,
    ):
        between = body[pos : m.start()]
        if between:
            # Anything sitting between runs would break the batching, so it
            # ends the current <w:del> and is re-emitted untouched.
            flush()
            out.append(between)
        pos = m.end()
        chunk = m.group(0)
        if chunk.startswith("<w:hyperlink"):
            flush()
            inner = re.sub(
                RUN_RE, lambda r: wrap(to_del_text(r.group(0))), chunk
            )
            out.append(inner)
        else:
            batch.append(to_del_text(chunk))
    flush()
    tail = body[pos:]
    if tail.strip():
        raise SystemExit(f"unhandled trailing paragraph content: {tail[:80]!r}")
    out.append(tail)

    # Delete the paragraph mark too, so the entry does not leave a blank line.
    mark = f'<w:rPr><w:del w:id="{_next_id()}" w:author="{AUTHOR}" w:date="{DATE}"/></w:rPr>'
    if "<w:pPr>" in head:
        head = head.replace("</w:pPr>", f"{mark}</w:pPr>")
    else:
        head = head + f"<w:pPr>{mark}</w:pPr>"

    new_para = head + "".join(out) + "</w:p>"
    return xml[: p_m.start()] + new_para + xml[p_m.end() :]


# ---------------------------------------------------------------------------
# Edit definitions
# ---------------------------------------------------------------------------

# Merges run first; each (anchor, count) collapses runs that a target sentence
# straddles so that the sentence lives in a single <w:t>.
MERGES = [
    ("although the latter reflects same-occasion assessment", 4),
    (", 2011). The NB dispersion parameter was estimated", 4),
    ("operates on the probability of reporting any crisis symptoms", 3),
]

ABSTRACT_OLD = (
    " Notably, in a secondary two-part analysis, semantic dimensions differed "
    "in whether they were associated with the occurrence of a crisis day or "
    "with its severity once symptoms were present."
)
ABSTRACT_NEW = (
    " In a secondary hurdle analysis, the association for one between-person "
    "component was confined to the probability of reporting any crisis "
    "symptoms rather than to severity on days when symptoms were present, "
    "suggesting that semantic features may relate to symptom occurrence and "
    "symptom intensity separately."
)

METHODS_OLD = (
    "Because 32.2% of person-days scored zero, we additionally compared this "
    "specification against zero-inflated negative binomial and negative "
    "binomial hurdle alternatives (see Tables S6 and S7). Both two-part models "
    "fit better by AIC and BIC (ΔAIC = 1070 and 1060, respectively) and "
    "reproduced the observed number of zeros more closely than the negative "
    "binomial, which under-predicted them (561 expected versus 809 observed)."
)
METHODS_NEW = (
    "Because 32.2% of person-days scored zero, we additionally compared this "
    "specification against zero-inflated negative binomial and negative "
    "binomial hurdle alternatives, both of which fit better by AIC and BIC and "
    "reproduced the observed number of zeros more closely than the negative "
    "binomial (Table S6)."
)

REPLACEMENTS = [
    ("abstract: occurrence versus severity", ABSTRACT_OLD, ABSTRACT_NEW),
    ("methods: model-comparison trim", METHODS_OLD, METHODS_NEW),
    ("methods: dash before 'that is'", " – that is, ", ", because "),
    (
        "methods: dash before 'a distinction'",
        "(p = .549) - a distinction the single-equation",
        "(p = .549), a distinction that the single-equation",
    ),
    (
        "references: Cohen 2024 journal title case",
        "The Journal of clinical psychiatry",
        "The Journal of Clinical Psychiatry",
    ),
    (
        "references: Cohen 2024 duplicated page range",
        "(4), 24m15320-24m15320.",
        "(4), 24m15320.",
    ),
]

S6_NOTE_ANCHOR = (
    "Expected zeros are model-implied counts of zero outcomes. NB = negative "
    "binomial; ZINB = zero-inflated negative binomial."
)
S6_NOTE_ADD = (
    " Relative to the negative binomial, AIC was lower by 1,070 for the hurdle "
    "model and by 1,060 for the zero-inflated negative binomial with covariate "
    "inflation."
)

DELETE_ANCHOR = (
    "The Narrative‐Crisis Model of suicide and its prediction of "
    "near‐term suicide risk."
)


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

    for anchor, count in MERGES:
        xml = merge_runs_in_para(xml, anchor, count)

    for label, old, new in REPLACEMENTS:
        try:
            xml = replace_in_run(xml, old, new)
        except SystemExit as exc:
            raise SystemExit(f"[{label}] {exc}") from exc

    xml = insert_after_in_run(xml, S6_NOTE_ANCHOR, S6_NOTE_ADD)
    xml = delete_paragraph(xml, DELETE_ANCHOR)

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
    new_ins = n_ins - before_ins
    new_del = n_del - before_del
    # Six replacements plus one pure insertion.
    assert new_ins == len(REPLACEMENTS) + 1, f"new insertions = {new_ins}"
    # Six replacements, plus two run groups and one paragraph mark for the
    # deleted reference entry.
    assert new_del == len(REPLACEMENTS) + 3, f"new deletions = {new_del}"

    authors = set(re.findall(r'w:author="([^"]*)"', xml))
    assert authors == {V12_AUTHOR, AUTHOR}, f"unexpected revision authors: {authors}"

    for label, _, new in REPLACEMENTS:
        assert esc(new) in xml, f"[{label}] inserted text missing"
    assert esc(S6_NOTE_ADD) in xml, "Table S6 note addition missing"

    # The deleted entry must survive only as delText.
    assert esc(DELETE_ANCHOR) not in re.sub(
        r"<w:delText[^>]*>.*?</w:delText>", "", xml, flags=re.S
    ), "Cohen (2022) entry still present as live text"

    # No OOXML tag may ever appear as literal document text.
    assert "&lt;w:" not in xml, "literal markup leaked into document text"

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
    print(f"  v12 revisions carried through: {before_ins} ins / {before_del} del")
    print(f"  new revisions: {new_ins} ins / {new_del} del, author {AUTHOR!r}")
    print("  0 tracked markup inside any table, 0 literal markup in text")


if __name__ == "__main__":
    main()
