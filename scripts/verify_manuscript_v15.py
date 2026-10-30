"""Check NLP_Crisis_EMA Paper v15.docx is exactly v14 with every change accepted.

The one check that matters: v15's paragraph text must equal v14's text under an
independently computed accept-all, paragraph for paragraph, allowing only for
the one paragraph whose mark was deleted and which therefore disappears.

v15 was produced by string surgery. This file resolves v14 with ElementTree
instead, reusing the accept_all already written for the v13 verifier, so the
flattening is checked against a different implementation rather than against
itself. That matters more here than usual: the v15 editor's first two drafts
both failed on tag-prefix collisions, <w:del ...> matching the self-closing
<w:del .../> and the substring "<w:del" matching <w:delText>.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from verify_manuscript_v13 import accept_all, load, para_texts, q

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKED = "NLP_Crisis_EMA Paper v14.docx"
DEFAULT_CLEAN = "NLP_Crisis_EMA Paper v15.docx"

R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

MUST_CONTAIN = [
    # The v12 analysis layer.
    "Fourth, Aim 1 was designed to establish convergent validity",
    "Fifth, the prospective associations were carried largely by stable",
    "Sixth, written diary summaries were collected only during the evening",
    "diary text appears to function as a convergent but noisier index",
    # The v13 and v14 copyedit layer.
    "In a secondary hurdle analysis, the association for one between-person",
    "hurdle alternatives, both of which fit better by AIC and BIC",
    "distinct latent subgroup, because a symptom-free day",
    "(p = .549), a distinction that the single-equation",
    "The Journal of Clinical Psychiatry",
    "85(4), 24m15320.",
    "Relative to the negative binomial, AIC was lower by 1,070",
    "413–422.",
    "https://doi.org/10.1111/sltb.12439",
    "https://doi.org/10.4088/JCP.24m15320",
    "https://doi.org/10.18653/v1/D19-1410",
]

MUST_NOT_CONTAIN = [
    "Notably, in a secondary two-part analysis",
    "(see Tables S6 and S7)",
    "ΔAIC = 1070 and 1060",
    "561 expected versus 809 observed",
    "– that is,",
    ".549) - a distinction",
    "The Journal of clinical psychiatry",
    "24m15320-24m15320",
    "The Narrative‐Crisis Model of suicide",
    "413-422.",
    "was an adequate proxy for crisis severity",
]

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def raw(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        return z.read("word/document.xml").decode("utf-8")


def relationships(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/_rels/document.xml.rels"))
    return {el.get("Id"): el.get("Target") for el in root}


def hyperlink_pairs(root: ET.Element) -> list[tuple[str, str]]:
    out = []
    for h in root.iter(q("hyperlink")):
        rid = h.get(f"{{{R}}}id")
        if rid:
            out.append((rid, "".join(t.text or "" for t in h.iter(q("t")))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracked", default=DEFAULT_TRACKED)
    ap.add_argument("--clean", default=DEFAULT_CLEAN)
    args = ap.parse_args()
    global V14, V15
    V14, V15 = ROOT / args.tracked, ROOT / args.clean

    print(f"{V14.name}  (all accepted)  ->  {V15.name}")

    for p in (V14, V15):
        if not p.exists():
            check(False, f"{p.name} present")
            sys.exit(1)

    with zipfile.ZipFile(V14) as a, zipfile.ZipFile(V15) as b:
        check(set(a.namelist()) == set(b.namelist()), "zip entry set unchanged")

    v14, v15 = load(V14), load(V15)

    # --- the central equivalence -------------------------------------------
    expected = [t for t in para_texts(accept_all(v14))]
    actual = para_texts(v15)

    # The withdrawn reference entry resolves to an empty paragraph under an
    # in-place accept, and to no paragraph at all in v15.
    trimmed = [t for i, t in enumerate(expected) if not (i == 110 and t == "")]
    check(len(expected) == 707 and len(actual) == 706,
          "paragraph count drops by exactly one",
          f"{len(expected)} -> {len(actual)}")
    check(expected[110] == "", "the dropped paragraph was empty after accepting",
          repr(expected[110][:40]))

    diff = [i for i, (x, y) in enumerate(zip(trimmed, actual)) if x != y]
    check(trimmed == actual,
          "v15 text equals an independent accept-all of v14",
          f"first differing paragraphs: {diff[:6]}" if diff else "706 paragraphs")

    # --- no revision markup survives ---------------------------------------
    x = raw(V15)
    for tag in ("ins", "del", "delText", "rPrChange", "pPrChange", "moveFrom",
                "moveTo", "cellIns", "cellDel"):
        check(not re.search(rf"<w:{tag}\b", x), f"no <w:{tag}> survives")
    check('w:author="' not in x, "no revision author survives")
    check('w:date="' not in x, "no revision date survives")
    check(not re.search(r"<w:hyperlink\b[^>]*>\s*</w:hyperlink>", x),
          "no empty hyperlink survives")

    # --- structure and links ------------------------------------------------
    check(len(list(v14.iter(q("tbl")))) == len(list(v15.iter(q("tbl")))),
          "table count unchanged", f"{len(list(v15.iter(q('tbl'))))} tables")
    check(
        sum(len(list(p.iter(q("tab")))) for p in v14.iter(q("p")))
        == sum(len(list(p.iter(q("tab")))) for p in v15.iter(q("p"))),
        "tab elements preserved",
    )

    rels = relationships(V15)
    pairs = hyperlink_pairs(v15)
    mismatched = [
        (rid, t, rels.get(rid)) for rid, t in pairs
        if t.startswith("http") and rels.get(rid) != t
    ]
    check(not mismatched, "every hyperlink label matches its relationship target",
          f"{len(pairs)} links" if not mismatched else str(mismatched[:3]))

    body = re.sub(r"\s+", " ", " ".join(actual))
    found = {t.rstrip(".") for t in re.findall(r"https://doi\.org/\S+", body)}
    live = {t.rstrip(".") for _, t in pairs}
    check(not (found - live), "every DOI is a live hyperlink",
          f"{len(found)} DOIs" if not found - live else f"plain: {sorted(found - live)}")

    # --- content spot checks ------------------------------------------------
    missing = [s for s in MUST_CONTAIN if re.sub(r"\s+", " ", s) not in body]
    check(not missing, "every accepted edit is present",
          f"missing: {[m[:40] for m in missing]}" if missing
          else f"{len(MUST_CONTAIN)} phrases")

    lingering = [s for s in MUST_NOT_CONTAIN if re.sub(r"\s+", " ", s) in body]
    check(not lingering, "no superseded wording is present",
          f"lingering: {lingering}" if lingering else f"{len(MUST_NOT_CONTAIN)} phrases")

    for ordinal in ("First,", "Second,", "Third,", "Fourth,", "Fifth,", "Sixth,"):
        if body.count(ordinal) < 1:
            check(False, f"limitation ordinal {ordinal} present")
            break
    else:
        check(True, "limitations read First through Sixth")

    check("<w:" not in body, "no literal OOXML tag in rendered text")

    print("all checks passed" if not failures else f"{len(failures)} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
