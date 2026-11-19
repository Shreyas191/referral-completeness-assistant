"""Sanity checks for case folders, run before a case is used for evaluation.

    python datagen/check_cases.py                      # every case under data/cases/
    python datagen/check_cases.py data/cases/hand-004  # one case

For every case:
  - every document loads and has a readable date
  - gold.json covers exactly the checklist's items
  - each gold evidence text appears verbatim on the cited page
  - present/stale labels agree with the item's recency window (same rule as pipeline/validator.py)
  - the identity check quarantines exactly the documents listed in wrong_patient_docs
  - no dose conflicts among kept documents, unless the case is in the contradiction slice
  - leak check: a missing or not-applicable item's description and synonyms appear nowhere in the patient's
    own documents (nor does a not-applicable item's condition word, e.g. "warfarin"), and a stale item's
    terms appear only in its evidence documents, since a newer mention would make it present

For the hand-made cases (issue #7), it also checks that the planted defect is the only place the item
shows up, so a correct answer cannot come from somewhere else in the packet. Exits 1 if anything fails.
"""
import re
import sys
from pathlib import Path

from referral.eval.gold import GoldCase, load_gold
from referral.ingest.chunker import chunk_case
from referral.ingest.identity import quarantine
from referral.ingest.pdf_loader import load_case
from referral.pipeline.conflicts import dose_conflicts
from referral.schemas import Checklist, Document

CASES = Path("data/cases")
CHECKLISTS = Path("data/checklists")


def term_hits(docs: list[Document], terms: list[str]) -> list[tuple[str, int]]:
    """(doc_id, page) for every whole-word, case-insensitive match of any term."""
    rx = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)
    return sorted({(d.doc_id, n) for d in docs for n, page in enumerate(d.pages, 1) if rx.search(page)})


def check_case(case_dir: Path) -> list[str]:
    docs, gold = load_case(case_dir), load_gold(case_dir)
    checklist = Checklist.model_validate_json((CHECKLISTS / f"{gold.checklist_id}.json").read_text())
    items = {i.item_id: i for i in checklist.items}
    by_id = {d.doc_id: d for d in docs}
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            problems.append(msg)

    check(all(d.doc_date for d in docs), f"unreadable date: {[d.doc_id for d in docs if not d.doc_date]}")
    check({i.item_id for i in gold.items} == set(items), "gold items differ from the checklist items")

    for gi in gold.items:
        for e in gi.evidence:
            doc = by_id.get(e.doc_id)
            found = doc is not None and 1 <= e.page <= len(doc.pages) and e.text in doc.pages[e.page - 1]
            check(found, f"{gi.item_id}: evidence not found verbatim on {e.doc_id} p{e.page}")
        window = items[gi.item_id].recency_days if gi.item_id in items else None
        dates = [by_id[e.doc_id].doc_date for e in gi.evidence if e.doc_id in by_id and by_id[e.doc_id].doc_date]
        if window and dates and gi.status.value in {"present", "stale"}:
            age = (gold.reference_date - max(dates)).days
            check((age > window) == (gi.status.value == "stale"),
                  f"{gi.item_id}: labelled {gi.status.value} but evidence is {age}d old (window {window}d)")

    kept, flagged = quarantine(docs)
    check(sorted(flagged) == sorted(gold.wrong_patient_docs),
          f"identity check quarantined {flagged}, gold lists {gold.wrong_patient_docs}")
    if "contradiction" not in gold.slices:
        conflicts = dose_conflicts(chunk_case(kept))
        check(not conflicts, f"unplanned dose conflicts: {conflicts}")

    problems += leaks(gold, docs, items)
    problems += check_planted(gold, docs, kept, items)
    return problems


def leaks(gold: GoldCase, docs: list[Document], items: dict) -> list[str]:
    """Places where an item the gold calls missing, not applicable or stale is mentioned anyway."""
    own = [d for d in docs if d.doc_id not in gold.wrong_patient_docs]  # foreign documents are quarantined
    found = []
    for gi in gold.items:
        item = items.get(gi.item_id)
        if item is None:
            continue
        hits = term_hits(own, [item.description, *item.synonyms])
        if gi.status.value in {"missing", "not_applicable"} and hits:
            found.append(f"leak: {gi.item_id} is {gi.status.value} but its terms appear on {hits}")
        if gi.status.value == "not_applicable" and item.condition:
            key_hits = term_hits(own, [item.condition.split()[-1]])
            if key_hits:
                found.append(f"leak: {gi.item_id} is not_applicable but its condition word appears on {key_hits}")
        if gi.status.value == "stale":
            newer = sorted({doc for doc, _ in hits} - {e.doc_id for e in gi.evidence})
            if newer:
                found.append(f"leak: {gi.item_id} is stale but also mentioned in {newer}")
    return found


def check_planted(gold: GoldCase, docs: list[Document], kept: list[Document], items: dict) -> list[str]:
    """Defect-specific checks for the hand-made cases."""
    problems = []
    cid = gold.case_id
    if cid == "hand-001":  # missing echo
        hits = term_hits(docs, items["echo"].synonyms + ["echocardiogram"])
        if hits:
            problems.append(f"echo terms leak into a missing-echo case: {hits}")
    elif cid == "hand-002":  # stale BMP
        hits = term_hits(docs, items["bmp"].synonyms + ["basic metabolic"])
        if hits != [("lab_report_1", 1)]:
            problems.append(f"BMP terms outside the old lab report: {hits}")
    elif cid == "hand-003":  # metoprolol dose contradiction
        conflicts = dose_conflicts(chunk_case(kept))
        docs_in_conflict = {doc for _, doc in conflicts.get("tartrate", set())}
        if set(conflicts) != {"tartrate"} or docs_in_conflict != {"medication_list", "progress_note"}:
            problems.append(f"expected only the metoprolol conflict between med list and note: {conflicts}")
    elif cid == "hand-004":  # ECG buried on page 11
        hits = term_hits(docs, items["ecg"].synonyms + ["electrocardiogram"])
        if hits != [("progress_note", 11)]:
            problems.append(f"ECG terms outside progress_note p11: {hits}")
    return problems


def main(argv: list[str]) -> int:
    case_dirs = [Path(a) for a in argv] or sorted(p for p in CASES.iterdir() if (p / "gold.json").exists())
    failed = 0
    for case_dir in case_dirs:
        problems = check_case(case_dir)
        print(f"{'ok  ' if not problems else 'FAIL'} {case_dir.name}")
        for p in problems:
            print(f"     - {p}")
        failed += bool(problems)
    print(f"\n{len(case_dirs) - failed}/{len(case_dirs)} cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
