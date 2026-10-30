"""Produce v15: v14 with every tracked change accepted and no revision markup.

v15 is the clean submission copy. v14 remains the reviewable record of what
changed from v10, so this script is a one-way flattening and nothing in it
needs to be reversible.

Accepting a change means, per paragraph:

  - every <w:del> subtree is removed, taking its <w:delText> with it;
  - every <w:ins> element is unwrapped, leaving its runs in place;
  - a paragraph whose paragraph mark is itself deleted merges into the one
    after it, which for the single such paragraph here (the withdrawn Cohen
    2022 entry, whose content is also fully deleted) means the paragraph
    disappears, so the count drops from 707 to 706.

The script refuses to guess at the general merge case. If a paragraph with a
deleted mark still has surviving text, it stops rather than inventing a
placement for that text.

Two cleanups follow from the above rather than being cosmetic. A <w:hyperlink>
whose only child was a deletion is left empty and is removed, which is what
strands relationship rId15; an unused external relationship is harmless, so it
stays rather than being renumbered. And a <w:rPr> inside <w:pPr> that held only
the paragraph-mark deletion is removed once empty.

Done by string surgery rather than ElementTree on purpose. ElementTree rewrites
namespace prefixes on output, and this document's mc:Ignorable attribute names
prefixes that must stay declared; a round trip through ElementTree risks
dropping them. Verification lives in verify_manuscript_v15.py, which does parse
with ElementTree and shares no code with this file.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "NLP_Crisis_EMA Paper v14.docx"
DST = ROOT / "NLP_Crisis_EMA Paper v15.docx"
WORK = Path("/tmp/v15work")

PARA_RE = re.compile(r"<w:p\b(?:(?!</w:p>).)*</w:p>", re.S)
HEAD_RE = re.compile(r"<w:p\b[^>]*>(?:<w:pPr>.*?</w:pPr>)?", re.S)
# A text element. The mandatory whitespace before attributes keeps <w:tab/> out.
T_RE = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.S)

# The trailing (?<!/) is load-bearing. Without it, <w:del ...> also matches the
# self-closing <w:del .../> that marks a deleted paragraph mark, and the lazy
# .*? then runs on to the next </w:del>, swallowing a whole unrelated deletion
# and the live text between them. Same hazard as the <w:tab/> trap in the v12
# editor: a pattern that looks like it matches an open tag also matches an
# empty element.
DEL_PAIR_RE = re.compile(r"<w:del\b[^>]*(?<!/)>.*?</w:del>", re.S)
DEL_EMPTY_RE = re.compile(r"<w:del\b[^>]*/>")
INS_OPEN_RE = re.compile(r"<w:ins\b[^>]*(?<!/)>")
# An inserted paragraph mark is <w:ins .../> inside pPr/rPr. Accepting it keeps
# the paragraph break, so the marker is simply dropped.
INS_EMPTY_RE = re.compile(r"<w:ins\b[^>]*/>")
INS_CLOSE_RE = re.compile(r"</w:ins>")
EMPTY_LINK_RE = re.compile(r"<w:hyperlink\b[^>]*>\s*</w:hyperlink>")
EMPTY_RPR_RE = re.compile(r"<w:rPr>\s*</w:rPr>")


def text_of(fragment: str) -> str:
    return "".join(T_RE.findall(fragment))


def accept(fragment: str) -> str:
    """Resolve every revision in a fragment that contains no nesting.

    Empty elements are cleared before paired ones so that a self-closing tag
    can never be mistaken for the opening half of a pair.
    """
    out = DEL_EMPTY_RE.sub("", fragment)
    out = DEL_PAIR_RE.sub("", out)
    out = INS_EMPTY_RE.sub("", out)
    out = INS_OPEN_RE.sub("", out)
    out = INS_CLOSE_RE.sub("", out)
    out = EMPTY_LINK_RE.sub("", out)
    return out


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(SRC) as z:
        names = z.namelist()
        z.extractall(WORK)

    doc = WORK / "word" / "document.xml"
    xml = doc.read_text(encoding="utf-8")

    before_paras = len(PARA_RE.findall(xml))
    before_tbl = len(re.findall(r"<w:tbl>", xml))
    before_tab = len(re.findall(r"<w:tab/>", xml))
    before_ins = len(re.findall(r"<w:ins\b", xml))
    before_del = len(re.findall(r"<w:del\b", xml))

    # Nothing below is safe if revisions nest, so establish that first. Each
    # match's inner content is everything between its own open and close tags.
    #
    # The \b in the search is load-bearing for the same reason the lookbehind
    # above is. A plain "<w:del" substring also matches <w:delText>, and a
    # plain "<w:ins" also matches <w:insideH> and <w:insideV> in table
    # borders, so both checks would fire on every ordinary deletion and every
    # bordered table.
    def inner(match: re.Match[str], tag: str) -> str:
        body = match.group(0)
        return body[body.index(">") + 1 : -len(f"</w:{tag}>")]

    for m in re.finditer(r"<w:ins\b[^>]*(?<!/)>.*?</w:ins>", xml, re.S):
        if re.search(r"<w:ins\b", inner(m, "ins")):
            raise SystemExit("nested tracked insertion; cannot flatten by pattern")
    for m in DEL_PAIR_RE.finditer(xml):
        if re.search(r"<w:del\b", inner(m, "del")):
            raise SystemExit("nested tracked deletion; cannot flatten by pattern")

    # Only these revision kinds are handled. Anything else must stop the run.
    for tag in (
        "rPrChange", "pPrChange", "tblPrChange", "tcPrChange", "trPrChange",
        "sectPrChange", "numberingChange", "cellIns", "cellDel", "cellMerge",
        "moveFrom", "moveTo",
    ):
        if re.search(rf"<w:{tag}\b", xml):
            raise SystemExit(f"unhandled revision markup <w:{tag}>")

    out: list[str] = []
    pos = 0
    dropped: list[int] = []

    for i, m in enumerate(PARA_RE.finditer(xml)):
        out.append(xml[pos : m.start()])
        pos = m.end()
        para = m.group(0)

        hm = HEAD_RE.match(para)
        if not hm:
            raise SystemExit(f"could not parse paragraph {i}")
        mark_deleted = bool(re.search(r"<w:del\b", hm.group(0)))

        resolved = accept(para)
        if mark_deleted:
            if text_of(resolved).strip():
                raise SystemExit(
                    f"paragraph {i} has a deleted mark but surviving text; "
                    "merging into the next paragraph is not implemented"
                )
            dropped.append(i)
            continue

        out.append(resolved)

    out.append(xml[pos:])
    xml = "".join(out)
    xml = EMPTY_RPR_RE.sub("", xml)

    # --- validation ---------------------------------------------------------
    import xml.dom.minidom as md

    md.parseString(xml.encode("utf-8"))

    assert not re.search(r"<w:ins\b", xml), "insertions survived"
    assert not re.search(r"<w:del\b", xml), "deletions survived"
    assert "<w:delText" not in xml, "delText survived"
    assert 'w:author="' not in xml, "a revision author survived"

    after_paras = len(PARA_RE.findall(xml))
    assert after_paras == before_paras - len(dropped), (
        f"paragraph count {after_paras}, expected {before_paras - len(dropped)}"
    )
    assert len(re.findall(r"<w:tbl>", xml)) == before_tbl, "table count changed"
    assert len(re.findall(r"<w:tab/>", xml)) == before_tab, "tab count changed"
    assert not EMPTY_LINK_RE.search(xml), "an empty hyperlink survived"
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
    print(f"  accepted {before_ins} insertions and {before_del} deletions")
    print(f"  paragraphs {before_paras} -> {after_paras}, dropped {dropped}")
    print(f"  tables {before_tbl}, tabs {before_tab}")
    print("  0 revision markup, 0 revision authors, 0 empty hyperlinks")


if __name__ == "__main__":
    main()
