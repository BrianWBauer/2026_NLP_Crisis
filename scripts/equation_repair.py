"""Repair the display equation's OMML in the Methods section.

The equation as authored has three empty Office Math elements, which Word draws
as dotted placeholder boxes:

  - <m:e/> on the <m:nary>, meaning the summation has an empty body. The whole
    summand sat outside the sigma as sibling elements, so the sum had nothing
    to sum over. This is the box immediately after the sigma.
  - two <m:sSub> wrappers whose <m:sub/> is empty, producing the box after
    "between,i" and the one at the end of the line.

build() re-emits the same equation with the summand inside <m:nary><m:e>, and
with no empty elements anywhere. The visible tokens are unchanged, which patch()
asserts by comparing the <m:t> sequence before and after, so this is a structural
repair and not a change to the notation.

This repair is NOT tracked. It changes no text, so there is nothing for a
tracked deletion or insertion to carry, and Word handles revision markup around
OMML poorly. It is applied to both the tracked copy and the clean copy so the
two do not disagree.

The defect is present in v10 and was not introduced by any edit in this repo.
"""

import re

RPR = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'
CTRL = f"<m:ctrlPr>{RPR}</m:ctrlPr>"


def r(t: str) -> str:
    return f'<m:r>{RPR}<m:t xml:space="preserve">{t}</m:t></m:r>'


def ssub(base: str, sub: str) -> str:
    return f"<m:sSub><m:sSubPr>{CTRL}</m:sSubPr><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"


def ssubsup(base: str, sub: str, sup: str) -> str:
    return (f"<m:sSubSup><m:sSubSupPr>{CTRL}</m:sSubSupPr><m:e>{base}</m:e>"
            f"<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>")


def build() -> str:
    beta_w = ssubsup(r("β"), r("k"), r("W"))
    beta_b = ssubsup(r("β"), r("k"), r("B"))
    summand = (
        r("(") + beta_w + r("PC ") + ssub(r("k"), r("within,it"))
        + r("+") + beta_b + r("PC ") + ssub(r("k"), r("between,i")) + r(")")
    )
    nary = (
        f'<m:nary><m:naryPr><m:chr m:val="∑"/>{CTRL}</m:naryPr>'
        f'<m:sub>{r("k = 1")}</m:sub><m:sup>{r("5")}</m:sup>'
        f"<m:e>{summand}</m:e></m:nary>"
    )
    log_box = (
        f'<m:box><m:boxPr><m:opEmu m:val="1"/>{CTRL}</m:boxPr>'
        f'<m:e>{r("log")}</m:e></m:box>'
    )
    body = (
        log_box + r("(E[") + ssub(r("y"), r("it")) + r("]) =")
        + ssub(r("β"), r("0")) + r("+") + nary
    )
    return f"<m:oMathPara><m:oMath>{body}</m:oMath></m:oMathPara>"


OMATH_RE = re.compile(r"<m:oMathPara>.*?</m:oMathPara>", re.S)


def patch(xml: str) -> str:
    hits = OMATH_RE.findall(xml)
    if len(hits) != 1:
        raise SystemExit(f"expected exactly 1 display equation, found {len(hits)}")
    old = hits[0]
    for empty in ("<m:e/>", "<m:sub/>"):
        if empty not in old:
            raise SystemExit(f"expected {empty} in the equation; already repaired?")
    new = build()
    for empty in ("<m:e/>", "<m:sub/>", "<m:sup/>", "<m:den/>", "<m:num/>"):
        if empty in new:
            raise SystemExit(f"rebuilt equation still contains {empty}")
    # Every visible token must survive the rebuild.
    def toks(s):
        return [t for t in re.findall(r"<m:t[^>]*>(.*?)</m:t>", s, re.S)]
    if toks(old) != toks(new):
        raise SystemExit(f"token mismatch\n old={toks(old)}\n new={toks(new)}")
    return xml.replace(old, new)
