"""Produce v14 of the manuscript from v13 by adding two verified DOIs.

Both DOIs were confirmed against Crossref:

  - Cohen et al. (2019), 10.1111/sltb.12439, where Crossref's 2018 date is the
    online-first date and volume 49, issue 2, pages 413 to 422 confirm the 2019
    issue the manuscript cites.
  - Cohen et al. (2024), 10.4088/JCP.24m15320, given in the publisher's own
    casing; Crossref returns the suffix lowercased and both resolve.

The 2019 entry also gets an en dash in its page range, matching the rest of the
reference list, which uses en dashes everywhere else.

These changes carry the same author and date as v13's, because they are the
same copyedit pass. Rejecting the "Copyedit revision" layer still lands exactly
on v12.

Three structural points, beyond the traps documented in make_manuscript_v13:

1. 38 of the 39 DOIs in this reference list are real hyperlinks rather than
   plain text, so these two are inserted as hyperlinks for consistency. That
   means writing word/_rels/document.xml.rels as well as document.xml.

2. <w:hyperlink> is EG_PContent and is not valid inside <w:ins>, exactly as it
   is not valid inside <w:del>. The nesting is therefore inverted: the
   hyperlink wraps the <w:ins>, not the other way round.

3. The 2024 entry's trailing text is itself a v13 tracked insertion, so the new
   content is appended after that <w:ins> as a sibling rather than edited into
   it. Nesting <w:ins> inside <w:ins> is markup Word handles inconsistently.

Helpers are imported from make_manuscript_v13 rather than copied, so there is
one implementation of the run-splitting regexes. Verification lives in
verify_manuscript_v14.py, which parses with ElementTree and shares no code with
either editor.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import make_manuscript_v13 as v13

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "NLP_Crisis_EMA Paper v13.docx"
DST = ROOT / "NLP_Crisis_EMA Paper v14.docx"
WORK = Path("/tmp/v14work")

AUTHOR = "Copyedit revision"
DATE = "2026-10-29T00:00:00Z"

# The imported helpers read these from their own module, so rebind them. The
# id range starts clear of v13's 20001 to 20012.
v13.AUTHOR = AUTHOR
v13.DATE = DATE
v13._id[0] = 21000

esc = v13.esc
replace_in_run = v13.replace_in_run
_find_para = v13._find_para

LINK_RPR = '<w:rPr><w:color w:val="1155CC"/><w:u w:val="single"/></w:rPr>'
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

DOIS = [
    # (paragraph anchor, DOI url, run properties for the inserted space)
    (
        "The suicidal narrative and its relationship to the suicide crisis",
        "https://doi.org/10.1111/sltb.12439",
    ),
    (
        "Diagnosis of the Suicide Crisis Syndrome in the Emergency Department",
        "https://doi.org/10.4088/JCP.24m15320",
    ),
]

EN_DASH_OLD = "(2), 413-422."
EN_DASH_NEW = "(2), 413–422."


def next_rel_id(rels: str) -> str:
    used = {int(m) for m in re.findall(r'Id="rId(\d+)"', rels)}
    n = max(used) + 1
    while n in used:
        n += 1
    return f"rId{n}"


def add_external_rel(rels: str, rid: str, target: str) -> str:
    """Append an external hyperlink relationship."""
    if f'Id="{rid}"' in rels:
        raise SystemExit(f"relationship {rid} already exists")
    entry = (
        f'<Relationship Id="{rid}" Type="{REL_NS}/hyperlink" '
        f'Target="{target}" TargetMode="External"/>'
    )
    if "</Relationships>" not in rels:
        raise SystemExit("could not find </Relationships>")
    return rels.replace("</Relationships>", entry + "</Relationships>")


def append_hyperlink(xml: str, anchor: str, rid: str, url: str) -> str:
    """Append a tracked space and a tracked hyperlink at the end of a paragraph.

    The paragraph's last run supplies the run properties for the space, so the
    inserted whitespace matches the surrounding text rather than the link.
    """
    p_m = _find_para(xml, anchor)
    para = p_m.group(0)

    runs = list(v13.RUN_RE.finditer(para))
    if not runs:
        raise SystemExit(f"paragraph has no runs: {anchor[:50]!r}")
    rpr_m = re.search(r"<w:rPr>.*?</w:rPr>", runs[-1].group(0), re.S)
    body_rpr = rpr_m.group(0) if rpr_m else ""

    v13._id[0] += 1
    space_id = v13._id[0]
    v13._id[0] += 1
    link_id = v13._id[0]

    addition = (
        f'<w:ins w:id="{space_id}" w:author="{AUTHOR}" w:date="{DATE}">'
        f'<w:r>{body_rpr}<w:t xml:space="preserve"> </w:t></w:r>'
        f"</w:ins>"
        f'<w:hyperlink r:id="{rid}">'
        f'<w:ins w:id="{link_id}" w:author="{AUTHOR}" w:date="{DATE}">'
        f"<w:r>{LINK_RPR}<w:t>{esc(url)}</w:t></w:r>"
        f"</w:ins>"
        f"</w:hyperlink>"
    )

    new_para = para[: -len("</w:p>")] + addition + "</w:p>"
    return xml[: p_m.start()] + new_para + xml[p_m.end() :]


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(SRC) as z:
        names = z.namelist()
        z.extractall(WORK)

    doc = WORK / "word" / "document.xml"
    relf = WORK / "word" / "_rels" / "document.xml.rels"
    xml = doc.read_text(encoding="utf-8")
    rels = relf.read_text(encoding="utf-8")

    before_paras = len(re.findall(r"<w:p\b", xml))
    before_tbl = len(re.findall(r"<w:tbl>", xml))
    before_tab = len(re.findall(r"<w:tab/>", xml))
    before_ins = len(re.findall(r"<w:ins\b", xml))
    before_del = len(re.findall(r"<w:del\b", xml))
    before_rels = len(re.findall(r"<Relationship\b", rels))

    # 1. En dash in the 2019 page range, matching the rest of the list.
    xml = replace_in_run(xml, EN_DASH_OLD, EN_DASH_NEW)

    # 2. The two DOI hyperlinks.
    for anchor, url in DOIS:
        rid = next_rel_id(rels)
        rels = add_external_rel(rels, rid, url)
        xml = append_hyperlink(xml, anchor, rid, url)

    # --- validation ---------------------------------------------------------
    import xml.dom.minidom as md

    md.parseString(xml.encode("utf-8"))
    md.parseString(rels.encode("utf-8"))

    assert len(re.findall(r"<w:p\b", xml)) == before_paras, "paragraph count changed"
    assert len(re.findall(r"<w:tbl>", xml)) == before_tbl, "table count changed"
    assert len(re.findall(r"<w:tab/>", xml)) == before_tab, "tab count changed"

    for t in re.findall(r"<w:tbl>.*?</w:tbl>", xml, re.S):
        assert not re.search(r"<w:(ins|del)\b", t), "tracked markup inside a table"

    new_ins = len(re.findall(r"<w:ins\b", xml)) - before_ins
    new_del = len(re.findall(r"<w:del\b", xml)) - before_del
    # One replacement (1 ins + 1 del) plus a space and a link per DOI.
    assert new_ins == 1 + 2 * len(DOIS), f"new insertions = {new_ins}"
    assert new_del == 1, f"new deletions = {new_del}"

    assert len(re.findall(r"<Relationship\b", rels)) == before_rels + len(DOIS), (
        "relationship count wrong"
    )

    # No <w:ins> may contain another, and none may contain a hyperlink.
    for m in re.finditer(r"<w:ins\b[^>]*>(.*?)</w:ins>", xml, re.S):
        assert "<w:ins" not in m.group(1), "nested tracked insertion"
        assert "<w:hyperlink" not in m.group(1), "hyperlink inside w:ins"
    for m in re.finditer(r"<w:del\b[^>]*>(.*?)</w:del>", xml, re.S):
        assert "<w:hyperlink" not in m.group(1), "hyperlink inside w:del"

    for _, url in DOIS:
        assert esc(url) in xml, f"DOI text missing: {url}"
        assert f'Target="{url}"' in rels, f"DOI relationship missing: {url}"
    assert esc(EN_DASH_NEW) in xml, "en dash page range missing"

    assert "&lt;w:" not in xml, "literal markup leaked into document text"

    doc.write_text(xml, encoding="utf-8")
    relf.write_text(rels, encoding="utf-8")

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
        md.parseString(z.read("word/_rels/document.xml.rels"))

    print(f"wrote {DST}")
    print(f"  paragraphs {before_paras}, tables {before_tbl}, tabs {before_tab}")
    print(f"  new revisions: {new_ins} ins / {new_del} del, author {AUTHOR!r}")
    print(f"  relationships {before_rels} -> {before_rels + len(DOIS)}")
    print("  0 nested insertions, 0 hyperlinks inside tracked markup")


if __name__ == "__main__":
    main()
