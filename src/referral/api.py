"""FastAPI backend for the review UI.

    uvicorn referral.api:app --reload
"""
import tempfile
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from referral.eval.harness import CHECKLISTS, load_checklist
from referral.ingest.pdf_loader import load_case
from referral.llm import LLM
from referral.pipeline.run import run_case
from referral.schemas import CaseResult

app = FastAPI(title="Referral Completeness Assistant")


@app.get("/checklists")
def list_checklists() -> list[str]:
    return sorted(p.stem for p in CHECKLISTS.glob("*.json"))


@app.post("/cases", response_model=CaseResult)
async def check_case(
    checklist_id: str = Form(...),
    referral_date: date = Form(...),
    files: list[UploadFile] = File(...),
) -> CaseResult:
    try:
        checklist = load_checklist(checklist_id)
    except FileNotFoundError:
        raise HTTPException(404, f"unknown checklist {checklist_id}")
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = Path(tmp) / "uploaded-case"
        case_dir.mkdir()
        for f in files:
            (case_dir / Path(f.filename).name).write_bytes(await f.read())
        docs = load_case(case_dir)
        if not docs:
            raise HTTPException(400, "no PDFs uploaded")
        return run_case(docs, checklist, referral_date, LLM())
