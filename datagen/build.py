"""Steps 4-6 of the generator: noise, PDFs, gold.json and the leak check, in one command.

    python -m datagen.build --checklist cardiology-v1 --n 5 --seed 0 --split dev
    python -m datagen.build --checklist cardiology-v1 --n 40 --seed 10 --split dev --quota   # the dev set
    python -m datagen.build --checklist cardiology-v1 --n 20 --seed 30 --split dev --quota --profile hard \
        --out data/hard_cases                                                                # E3 / E4 set

For each plan (same case ids and seeds as `python -m datagen.render`): reuse
data/rendered/<case_id>/rendered.json when it exists, otherwise render the case with the LLM; add OCR-style
noise; lay the documents out as PDFs; read them back with the pipeline's own loader and write gold.json from
what the loader actually reads; then run datagen/check_cases.py on the folder (verbatim evidence, recency,
identity, dose conflicts, leaks) and print every flagged case.

Noise policy, chosen so the labels stay true:
  - letters only: look-alike swaps (rn<->m, l<->I, cl->d, ...), a dropped or doubled letter, or two words run
    together; only words of 4+ letters with no digits, so values, doses, units and short abbreviations never
    change
  - the first paragraph of every page (title, date, name, DOB, MRN, or the running header) is never touched,
    so document dates and patient identities parse exactly as rendered
  - evidence is noised like any other text, and gold.json stores it as the loader reads it
Page headers and footers carry no dates, because the loader takes a document's date from the first date
near its top.
"""
import argparse
import json
import random
import re
import textwrap
from pathlib import Path

from datagen.check_cases import check_case
from datagen.defect_plan import MIX, MIX_HARD, make_plan, select_plans
from datagen.render import LeakError, RenderedCase, render_case, write_case
from referral.eval.gold import GoldCase, GoldEvidence, GoldItem
from referral.ingest.pdf_loader import guess_doc_date, load_case
from referral.schemas import Checklist

NOISE_RATE = 0.02  # chance that an eligible word is corrupted
SLICE_BY_DEFECT = {"omit": "missing", "contradict": "contradiction", "stale": "stale",
                   "bury": "buried", "abbreviate": "abbreviation"}

# ---------------------------------------------------------------- noise

TOKEN_RE = re.compile(r"\s+|\S+")
NOISY_WORD = re.compile(r"^[A-Za-z][A-Za-z'\-]{3,}[.,;:)]?$")
LOOKALIKES = [("rn", "m"), ("m", "rn"), ("cl", "d"), ("l", "I"), ("I", "l"), ("e", "c"), ("h", "b"), ("ri", "n")]


def _corrupt(word: str, op: str, rng: random.Random) -> str:
    if op == "swap":
        pairs = [(a, b) for a, b in LOOKALIKES if a in word]
        if pairs:
            a, b = rng.choice(pairs)
            i = rng.choice([m.start() for m in re.finditer(re.escape(a), word)])
            return word[:i] + b + word[i + len(a):]
        op = "drop"
    interior = [i for i, c in enumerate(word) if c.isalpha()][1:-1]
    if not interior:
        return word
    i = rng.choice(interior)
    return word[:i] + word[i + 1:] if op == "drop" else word[:i] + word[i] + word[i:]


def add_noise(page: str, spans: list[tuple[int, int]], rate: float, rng: random.Random):
    """Corrupt eligible words; return the new page and where each (start, end) span now lies.

    Spans must start and end on word boundaries (evidence is whole lines). Newlines are never removed, so
    paragraph and line structure survive.
    """
    protect = page.find("\n\n")
    protect = len(page) if protect < 0 else protect
    tokens = [(m.start(), m.group()) for m in TOKEN_RE.finditer(page)]
    out, length, where = [], 0, {}
    skip_space = False
    for k, (start, tok) in enumerate(tokens):
        where[start] = length
        if skip_space:  # this space was removed to run two words together
            skip_space = False
            continue
        new = tok
        if start >= protect and NOISY_WORD.match(tok) and rng.random() < rate:
            op = rng.choice(["swap", "swap", "drop", "double", "merge"])
            if op == "merge":
                if k + 2 < len(tokens) and tokens[k + 1][1] == " ":
                    skip_space = True
            else:
                new = _corrupt(tok, op, rng)
        out.append(new)
        length += len(new)
    where[len(page)] = length
    return "".join(out), [(where[s], where[e]) for s, e in spans]


# ---------------------------------------------------------------- PDF layout

FONT, SIZE, LEADING = "Courier", 9, 11
PAGE_W, PAGE_H, MARGIN = 612, 792, 54  # US letter, in points
WRAP = 90                               # Courier 9 pt is 5.4 pt per character: 90 chars fit in 504 pt
LINES_PER_PAGE = (PAGE_H - 2 * MARGIN - 4 * LEADING) // LEADING


def layout(pages: list[str]) -> list[list[str]]:
    """Wrap and paginate. Each rendered page starts a new PDF page; a paragraph that does not fit on the rest of
    a page moves to the next one whole, so evidence (always whole paragraphs or lines) is never split across
    pages unless a single paragraph is longer than a page."""
    out: list[list[str]] = []
    for page in pages:
        current: list[str] = []
        for para in page.strip("\n").split("\n\n"):
            lines = [w for line in para.split("\n")
                     for w in (textwrap.wrap(line, WRAP, break_on_hyphens=False) or [""])]
            if current and len(current) + len(lines) > LINES_PER_PAGE:
                out.append(current)
                current = []
            while len(lines) > LINES_PER_PAGE:
                out.append(lines[:LINES_PER_PAGE])
                lines = lines[LINES_PER_PAGE:]
            current += lines + [""]
        out.append(current)
    return out


def write_pdf(path: Path, pdf_pages: list[list[str]], top: str) -> None:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H), invariant=1)  # invariant: byte-identical reruns
    for n, lines in enumerate(pdf_pages, 1):
        c.setFont(FONT, SIZE - 1)
        c.drawString(MARGIN, PAGE_H - MARGIN + LEADING, top)
        c.drawString(MARGIN, MARGIN - 2 * LEADING, f"Page {n} of {len(pdf_pages)}")
        c.setFont(FONT, SIZE)
        y = PAGE_H - MARGIN - LEADING
        for line in lines:
            c.drawString(MARGIN, y, line)
            y -= LEADING
        c.showPage()
    c.save()


# ---------------------------------------------------------------- gold from what the loader reads

def locate(needle: str, pages: list[str]) -> tuple[int, str] | None:
    """(page, exact text) of needle in pages, ignoring differences in whitespace (wrapping turns spaces into
    line breaks). The returned text is a verbatim substring of that page."""
    target = " ".join(needle.split())
    for n, page in enumerate(pages, 1):
        norm, index = [], []
        for i, ch in enumerate(page):
            if ch.isspace():
                if norm and norm[-1] != " ":
                    norm.append(" ")
                    index.append(i)
            else:
                norm.append(ch)
                index.append(i)
        at = "".join(norm).find(target)
        if at >= 0:
            return n, page[index[at]:index[at + len(target) - 1] + 1]
    return None


def build_case(case: RenderedCase, out_dir: Path, split: str, noise: float = NOISE_RATE, seed: int = 0) -> Path:
    """Write <out_dir>/<case_id>/ with one PDF per document and gold.json. Returns the folder."""
    if case.case_id.startswith("hand-"):
        raise ValueError("hand-* folders are hand-made and committed; generated cases must use other ids")
    case_dir = out_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for old in [*case_dir.glob("*.pdf"), *case_dir.glob("*.txt"), *case_dir.glob("gold.json")]:
        old.unlink()  # a rebuild replaces the generated files only

    rng = random.Random(f"noise:{seed}:{case.case_id}")
    noisy: dict[str, dict[tuple[int, int], str]] = {}  # doc_id -> {(item, n): evidence text after noise}
    for doc in case.docs:
        pages = []
        for n, page in enumerate(doc.pages, 1):
            ev = [(k, e) for k, e in enumerate(case.evidence) if e.doc_id == doc.doc_id and e.page == n]
            spans = [(page.index(e.text), page.index(e.text) + len(e.text)) for _, e in ev]
            new_page, new_spans = add_noise(page, spans, noise, rng)
            for (k, _), (s, t) in zip(ev, new_spans):
                noisy.setdefault(doc.doc_id, {})[k] = new_page[s:t]
            pages.append(new_page)
        write_pdf(case_dir / f"{doc.doc_id}.pdf", layout(pages), f"{case.patient.practice} | Confidential")

    loaded = {d.doc_id: d for d in load_case(case_dir)}
    for doc in case.docs:
        read_date = guess_doc_date(loaded[doc.doc_id].pages[0])
        if read_date != doc.doc_date:
            raise AssertionError(f"{case.case_id}/{doc.doc_id}: loader reads date {read_date}, rendered {doc.doc_date}")

    evidence: dict[str, list[GoldEvidence]] = {}
    for k, e in enumerate(case.evidence):
        found = locate(noisy[e.doc_id][k], loaded[e.doc_id].pages)
        if found is None:
            raise AssertionError(f"{case.case_id}: {e.item_id} evidence not found in {e.doc_id}.pdf")
        page, text = found
        evidence.setdefault(e.item_id, []).append(GoldEvidence(doc_id=e.doc_id, page=page, text=text))

    defects = {p.defect for p in case.plan.items if p.defect}
    slices = sorted({SLICE_BY_DEFECT[d] for d in defects} | ({"wrong_patient"} if case.wrong_patient_docs else set())
                    | (set() if defects else {"complete"}))
    gold = GoldCase(case_id=case.case_id, checklist_id=case.checklist_id, reference_date=case.reference_date,
                    split=split, slices=slices, wrong_patient_docs=case.wrong_patient_docs,
                    items=[GoldItem(item_id=i, status=s, evidence=evidence.get(i, []))
                           for i, s in case.expected.items()])
    (case_dir / "gold.json").write_text(gold.model_dump_json(indent=2))
    (case_dir / "rendered.json").write_text(case.model_dump_json(indent=2))  # lets anyone rebuild without the API
    return case_dir


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checklist", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", required=True, choices=["dev", "test", "adversarial"])
    ap.add_argument("--noise", type=float, default=NOISE_RATE, help="per-word corruption rate; 0 = clean text")
    ap.add_argument("--rendered", type=Path, default=Path("data/rendered"))
    ap.add_argument("--out", type=Path, default=Path("data/cases"))
    ap.add_argument("--quota", action="store_true",
                    help="pick the n plans to match the dataset mix (ids <checklist>-<split>-000...)")
    ap.add_argument("--profile", default="standard", choices=["standard", "hard"],
                    help="hard: ~45-50 page packets, evidence at random pages, more contradictions / stale / bury")
    ap.add_argument("--first", type=int, help="only process the first K selected cases (render in batches)")
    args = ap.parse_args()

    raw = json.loads(Path(f"data/checklists/{args.checklist}.json").read_text())
    checklist = Checklist.model_validate(raw)
    llm = None
    mix = MIX_HARD if args.profile == "hard" else MIX
    tag = "hard" if args.profile == "hard" else args.split
    if args.quota:
        plans, targets, achieved = select_plans(raw, args.n, args.seed, f"{args.checklist}-{tag}", mix=mix)
        print("mix    " + "  ".join(f"{k} {achieved[k]}/{targets[k]}" for k in targets) + "  (achieved/target)\n")
    else:
        rng = random.Random(args.seed)
        plans = [make_plan(f"{args.checklist}-{args.seed}-{n:03d}", raw, rng, mix) for n in range(args.n)]
    plans = plans[:args.first] if args.first else plans
    flagged = 0
    for plan in plans:
        cached = args.rendered / plan.case_id / "rendered.json"
        try:
            if cached.exists():
                case = RenderedCase.model_validate_json(cached.read_text())
                if case.plan != plan or case.profile != args.profile:
                    raise ValueError(f"{cached} was rendered from a different plan or profile; delete it to re-render")
                how = "reused"
            else:
                if llm is None:
                    from referral.llm import LLM
                    llm = LLM()
                case = render_case(plan, checklist, llm, seed=args.seed, profile=args.profile)
                write_case(case, args.rendered)
                how = f"rendered ({case.llm_calls} calls)"
            case_dir = build_case(case, args.out, args.split, args.noise, args.seed)
        except (LeakError, ValueError, AssertionError) as err:
            flagged += 1
            print(f"FAIL    {plan.case_id}: {err}")
            continue
        problems = check_case(case_dir)
        flagged += bool(problems)
        print(f"{'FLAGGED' if problems else 'ok     '} {plan.case_id} [{how}]")
        for p in problems:
            print(f"        - {p}")
    if llm is not None:
        print(f"\nLLM: {llm.calls} calls, {llm.input_tokens} input / {llm.output_tokens} output tokens")
    print(f"\n{len(plans) - flagged}/{len(plans)} cases built and passed every check "
          f"into {args.out}/ (split={args.split})")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
