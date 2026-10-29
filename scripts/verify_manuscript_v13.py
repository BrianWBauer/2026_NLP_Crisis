"""Check NLP_Crisis_EMA Paper v13.docx against v12.

Written to share no code with make_manuscript_v13.py. That script edits the XML
with regular expressions; this one parses it with ElementTree and walks the
element tree. A parsing bug in the editor therefore cannot hide behind the same
bug in the verifier, which is the failure mode that matters most here: the
naive ``<w:t[^>]*>`` pattern also matches ``<w:tab/>``, and if both files made
that mistake the errors would cancel and the run would report success.

The two reconstructions below are the substance of the check. Rejecting every
revision must reproduce v12's rendered text exactly, and accepting every
revision must produce the intended text, with each replaced phrase gone and
each new phrase present.
"""

from __future__ import annotations

import copy
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "NLP_Crisis_EMA Paper v12.docx"
DST = ROOT / "NLP_Crisis_EMA Paper v13.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def q(tag: str) -> str:
    return f"{{{W}}}{tag}"


V12_AUTHOR = "Analysis revision"
V13_AUTHOR = "Copyedit revision"

# Phrases that must disappear once the v13 revisions are accepted.
GONE = [
    "Notably, in a secondary two-part analysis",
    "(see Tables S6 and S7)",
    "ΔAIC = 1070 and 1060",
    "561 expected versus 809 observed",
    "– that is,",
    ".549) - a distinction",
    "The Journal of clinical psychiatry",
    "24m15320-24m15320",
    "The Narrative‐Crisis Model of suicide",
]

# Phrases that must appear once the v13 revisions are accepted.
PRESENT = [
    "In a secondary hurdle analysis, the association for one between-person "
    "component was confined to the probability of reporting any crisis "
    "symptoms rather than to severity on days when symptoms were present, "
    "suggesting that semantic features may relate to symptom occurrence and "
    "symptom intensity separately.",
    "hurdle alternatives, both of which fit better by AIC and BIC and "
    "reproduced the observed number of zeros more closely than the negative "
    "binomial (Table S6).",
    "does not correspond to a distinct latent subgroup, because a "
    "symptom-free day",
    "(p = .549), a distinction that the single-equation negative binomial "
    "necessarily collapses",
    "The Journal of Clinical Psychiatry",
    "85(4), 24m15320.",
    "Relative to the negative binomial, AIC was lower by 1,070 for the hurdle "
    "model and by 1,060 for the zero-inflated negative binomial with "
    "covariate inflation.",
]

# v12 content that must survive untouched.
V12_KEEP = [
    "Fourth, Aim 1 was designed to establish convergent validity",
    "Fifth, the prospective associations were carried largely by stable",
    "Sixth, written diary summaries were collected only during the evening",
]

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  {tag}  {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def load(path: Path) -> ET.Element:
    with zipfile.ZipFile(path) as z:
        return ET.fromstring(z.read("word/document.xml"))


def para_texts(root: ET.Element) -> list[str]:
    """Rendered text of every w:p, in document order."""
    out = []
    for p in root.iter(q("p")):
        out.append("".join(t.text or "" for t in p.iter(q("t"))))
    return out


def _mine(el: ET.Element, authors: set[str] | None) -> bool:
    """Whether this revision belongs to the author set being resolved."""
    return authors is None or (el.get(q("author")) or "") in authors


def _strip(
    parent: ET.Element, drop: str, unwrap: str, authors: set[str] | None
) -> None:
    """Depth-first: delete `drop` subtrees, splice `unwrap` children in place.

    Revisions by other authors are left standing, so one author's layer can be
    resolved while another's stays as tracked markup.
    """
    children = list(parent)
    new: list[ET.Element] = []
    for child in children:
        if child.tag == q(drop) and _mine(child, authors):
            continue
        if child.tag == q(unwrap) and _mine(child, authors):
            _strip(child, drop, unwrap, authors)
            if unwrap == "del":
                for dt in child.iter(q("delText")):
                    dt.tag = q("t")
            new.extend(list(child))
            continue
        _strip(child, drop, unwrap, authors)
        new.append(child)
    for child in children:
        parent.remove(child)
    for child in new:
        parent.append(child)


def reject(root: ET.Element, authors: set[str] | None = None) -> ET.Element:
    """The document as it reads with `authors`' revisions rejected."""
    r = copy.deepcopy(root)
    _strip(r, drop="ins", unwrap="del", authors=authors)
    # A deleted paragraph mark is restored on reject.
    for ppr in r.iter(q("pPr")):
        for rpr in ppr.findall(q("rPr")):
            for d in rpr.findall(q("del")):
                if _mine(d, authors):
                    rpr.remove(d)
    return r


def accept_all(root: ET.Element) -> ET.Element:
    """The document as it would read with every revision accepted."""
    r = copy.deepcopy(root)
    _strip(r, drop="del", unwrap="ins", authors=None)
    return r


def main() -> None:
    print(f"{SRC.name}  ->  {DST.name}")

    with zipfile.ZipFile(SRC) as a, zipfile.ZipFile(DST) as b:
        src_names, dst_names = set(a.namelist()), set(b.namelist())
    check(src_names == dst_names, "zip entry set unchanged")

    src = load(SRC)
    dst = load(DST)

    src_paras = para_texts(src)
    dst_paras = para_texts(dst)
    check(
        len(src_paras) == len(dst_paras),
        "paragraph count unchanged",
        f"{len(src_paras)} vs {len(dst_paras)}",
    )

    src_tbl = len(list(src.iter(q("tbl"))))
    dst_tbl = len(list(dst.iter(q("tbl"))))
    check(src_tbl == dst_tbl, "table count unchanged", f"{dst_tbl} tables")

    offending = 0
    for tbl in dst.iter(q("tbl")):
        for tag in ("ins", "del"):
            offending += len(list(tbl.iter(q(tag))))
    check(offending == 0, "no tracked markup inside any table", f"{offending} offending")

    authors = {
        el.get(q("author"))
        for tag in ("ins", "del")
        for el in dst.iter(q(tag))
        if el.get(q("author"))
    }
    check(
        authors == {V12_AUTHOR, V13_AUTHOR},
        "two revision authors",
        str(sorted(authors)),
    )

    counts: dict[tuple[str, str], int] = {}
    for tag in ("ins", "del"):
        for el in dst.iter(q(tag)):
            key = (el.get(q("author")) or "", tag)
            counts[key] = counts.get(key, 0) + 1
    check(
        counts.get((V12_AUTHOR, "ins")) == 5 and counts.get((V12_AUTHOR, "del")) == 5,
        "v12 revisions carried through unchanged",
        f"{counts.get((V12_AUTHOR, 'ins'))} ins / {counts.get((V12_AUTHOR, 'del'))} del",
    )
    check(
        counts.get((V13_AUTHOR, "ins")) == 7 and counts.get((V13_AUTHOR, "del")) == 9,
        "v13 revisions present",
        f"{counts.get((V13_AUTHOR, 'ins'))} ins / {counts.get((V13_AUTHOR, 'del'))} del",
    )

    # --- the three reconstructions -----------------------------------------
    # Rejecting only the new layer must land back on v12.
    rejected = para_texts(reject(dst, {V13_AUTHOR}))
    diff = [i for i, (a, b) in enumerate(zip(src_paras, rejected)) if a != b]
    check(
        rejected == src_paras,
        "REJECTING the v13 layer reproduces v12 exactly",
        f"paragraphs differing: {diff[:6]}" if diff else "",
    )

    # Rejecting both layers must land back on v10, the last clean manuscript.
    v10 = ROOT / "NLP_Crisis_EMA Paper v10.docx"
    if v10.exists():
        v10_paras = para_texts(load(v10))
        all_rejected = para_texts(reject(dst, None))
        diff10 = [
            i for i, (a, b) in enumerate(zip(v10_paras, all_rejected)) if a != b
        ]
        check(
            all_rejected == v10_paras,
            "REJECTING both layers reproduces v10 exactly",
            f"paragraphs differing: {diff10[:6]}" if diff10 else "",
        )
    else:
        check(False, "v10 available for the clean-baseline comparison")

    accepted = para_texts(accept_all(dst))
    body = " ".join(accepted)
    body_norm = re.sub(r"\s+", " ", body)

    missing = [p for p in PRESENT if re.sub(r"\s+", " ", p) not in body_norm]
    check(not missing, "accepted text contains every new phrase",
          f"missing: {[m[:40] for m in missing]}" if missing else "7 phrases")

    lingering = [p for p in GONE if re.sub(r"\s+", " ", p) in body_norm]
    check(not lingering, "accepted text contains no replaced phrase",
          f"lingering: {lingering}" if lingering else "9 phrases removed")

    kept = [p for p in V12_KEEP if re.sub(r"\s+", " ", p) not in body_norm]
    check(not kept, "v12 limitations survive in accepted text",
          f"missing: {kept}" if kept else "Fourth, Fifth, Sixth")

    # The deleted entry must still be recoverable on reject.
    rej_body = re.sub(r"\s+", " ", " ".join(rejected))
    check(
        "The Narrative‐Crisis Model of suicide" in rej_body,
        "deleted reference reappears when the change is rejected",
    )

    # It was an orphan: no in-text citation pointed at it.
    n_cite = len(re.findall(r"Cohen et al\., 2022", body_norm))
    check(n_cite == 0, "no in-text citation of Cohen et al. (2022) is left dangling",
          f"{n_cite} citations")

    # --- the numbers added to the Table S6 note ----------------------------
    # Recompute the AIC deltas from the table's own cells rather than trusting
    # the prose. Row order in Table S6 is hurdle, ZINB(cov), ZINB(int), NB.
    aics: dict[str, float] = {}
    for tbl in dst.iter(q("tbl")):
        rows = list(tbl.iter(q("tr")))
        cells = [
            ["".join(t.text or "" for t in c.iter(q("t"))).strip() for c in r.iter(q("tc"))]
            for r in rows
        ]
        if not any(c and "Log-likelihood" in " ".join(c) for c in cells):
            continue
        for row in cells:
            nums = [x for x in row if re.fullmatch(r"-?\d+\.\d+", x)]
            if row and len(nums) >= 3:
                aics[row[0]] = float(nums[1])
    check(
        len(aics) == 4,
        "Table S6 AIC column parsed",
        f"{len(aics)} rows: {sorted(aics)}",
    )
    if len(aics) == 4:
        nb = aics["NB"]
        d_hurdle = round(nb - aics["NB hurdle"])
        d_zinb = round(nb - aics["ZINB (covariate inflation)"])
        check(
            d_hurdle == 1070 and d_zinb == 1060,
            "note's AIC deltas match the table's own AIC column",
            f"hurdle {d_hurdle}, ZINB {d_zinb}",
        )

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
    check(
        changed == [20, 59, 88, 92, 96, 110, 111, 659],
        "changes confined to the expected paragraphs",
        str(changed),
    )

    print("all checks passed" if not failures else f"{len(failures)} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
