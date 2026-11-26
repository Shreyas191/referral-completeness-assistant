"""FastAPI backend for the review UI and the demo.

    uvicorn referral.api:app --reload

Endpoints:
  GET  /                            review page for intake staff (static/review.html)
  GET  /health                      liveness and the configured model
  GET  /checklists                  checklist ids
  GET  /checklists/{checklist_id}   one checklist (item descriptions, windows, definitions)
  GET  /samples                     stored synthetic cases the demo can run (dev, hand-written, hard; never test)
  POST /samples/{case_id}/check     run the pipeline on a stored case; returns the result, the gold labels and cost
  POST /cases                       run the pipeline on uploaded PDFs (multipart: checklist_id, referral_date, files)

Routes are plain functions, so FastAPI runs each in a worker thread and a 15-second pipeline call does not block
other requests. The LLM comes from `get_llm`, which tests override with a fake. The system never contacts anyone:
follow-ups are drafts in the response, for intake staff to review.
"""
import tempfile
import time
from datetime import date
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from referral.eval.gold import load_gold
from referral.eval.harness import CASES, CHECKLISTS, load_checklist
from referral.eval.report import cost_usd
from referral.ingest.pdf_loader import load_case
from referral.llm import LLM
from referral.pipeline.run import run_case
from referral.schemas import CaseResult, Checklist, Status

HARD_CASES = Path("data/hard_cases")
SAMPLE_DIRS = [CASES, HARD_CASES]  # data/test_cases is never listed: the held-out split stays untouched
UPLOAD_SUFFIXES = {".pdf", ".txt"}
PAGE = Path(__file__).parent / "static" / "review.html"

app = FastAPI(title="Referral Completeness Assistant")


def get_llm():
    return LLM()


class DocumentInfo(BaseModel):
    doc_id: str
    pages: int
    doc_date: date | None


class Sample(BaseModel):
    case_id: str
    split: str
    referral_date: date
    documents: int
    pages: int


class CheckResponse(BaseModel):
    result: CaseResult
    documents: list[DocumentInfo]
    gold: dict[str, Status] | None = None  # stored samples only, so a demo can show where the system agrees
    seconds: float
    llm_calls: int
    cost_usd: float | None


def _checklist(checklist_id: str):
    if checklist_id not in list_checklists():  # also stops path tricks like "../x"
        raise HTTPException(404, f"unknown checklist {checklist_id!r}")
    return load_checklist(checklist_id)


def _run(docs, checklist, referral_date: date, llm, gold=None) -> CheckResponse:
    calls, tin, tout = (getattr(llm, a, 0) for a in ("calls", "input_tokens", "output_tokens"))
    start = time.perf_counter()
    result = run_case(docs, checklist, referral_date, llm)
    tin, tout = getattr(llm, "input_tokens", 0) - tin, getattr(llm, "output_tokens", 0) - tout
    return CheckResponse(
        result=result,
        documents=[DocumentInfo(doc_id=d.doc_id, pages=len(d.pages), doc_date=d.doc_date) for d in docs],
        gold={gi.item_id: gi.status for gi in gold.items} if gold else None,
        seconds=round(time.perf_counter() - start, 2),
        llm_calls=getattr(llm, "calls", 0) - calls,
        cost_usd=cost_usd(tin, tout) if tin or tout else None,
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def review_page() -> str:
    return PAGE.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    from referral.config import settings

    return {"status": "ok", "model": f"{settings.llm_provider}:{settings.llm_model}"}


@app.get("/checklists")
def list_checklists() -> list[str]:
    return sorted(p.stem for p in CHECKLISTS.glob("*.json"))


@app.get("/checklists/{checklist_id}")
def get_checklist(checklist_id: str) -> Checklist:
    return _checklist(checklist_id)


def _sample_dirs() -> dict[str, Path]:
    out = {}
    for root in SAMPLE_DIRS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if (d / "gold.json").exists() and load_gold(d).split != "test":
                out[d.name] = d
    return out


def _page_counts(case_dir: Path) -> list[int]:
    """Pages per document without extracting text (listing 63 cases with full text extraction took 19 s)."""
    import pdfplumber

    counts = []
    for path in sorted(case_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            with pdfplumber.open(path) as pdf:
                counts.append(len(pdf.pages))
        elif path.suffix.lower() == ".txt":
            counts.append(len(path.read_text(encoding="utf-8").split("\f")))
    return counts


@lru_cache(maxsize=1)
def _samples() -> tuple[Sample, ...]:
    out = []
    for case_id, d in _sample_dirs().items():
        gold, pages = load_gold(d), _page_counts(d)
        out.append(Sample(case_id=case_id, split=gold.split, referral_date=gold.reference_date,
                          documents=len(pages), pages=sum(pages)))
    return tuple(out)


@app.get("/samples")
def list_samples() -> list[Sample]:
    return list(_samples())


@app.post("/samples/{case_id}/check")
def check_sample(case_id: str, llm=Depends(get_llm)) -> CheckResponse:
    d = _sample_dirs().get(case_id)  # only listed cases, so no test case and no path tricks
    if d is None:
        raise HTTPException(404, f"unknown sample {case_id!r}")
    gold = load_gold(d)
    return _run(load_case(d), _checklist(gold.checklist_id), gold.reference_date, llm, gold)


@app.post("/cases")
def check_case(
    checklist_id: str = Form(...),
    referral_date: date = Form(...),
    files: list[UploadFile] = File(...),
    llm=Depends(get_llm),
) -> CheckResponse:
    checklist = _checklist(checklist_id)
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = Path(tmp) / "uploaded-case"
        case_dir.mkdir()
        for f in files:
            name = Path(f.filename or "").name
            if Path(name).suffix.lower() not in UPLOAD_SUFFIXES:
                raise HTTPException(400, f"{name!r}: only .pdf and .txt documents are accepted")
            (case_dir / name).write_bytes(f.file.read())
        docs = load_case(case_dir)
        if not docs:
            raise HTTPException(400, "no documents uploaded")
        return _run(docs, checklist, referral_date, llm)
