"""The API end to end with the fake LLM: no key, no network."""
import json

import pytest
from fakes import FakeLLM
from fastapi.testclient import TestClient

from referral import api

HEADER = "Patient Name: Jane Doe\nDOB: 03/14/1961\nMRN: AB12345"
LETTER = f"Date: 11/01/2026\n{HEADER}\n\nReason for referral: exertional chest pain.\n\nCurrent medications: Metformin."
LABS = f"Date: 10/20/2026\n{HEADER}\n\nBMP: creatinine 1.1 mg/dL, potassium 4.2 mmol/L."


@pytest.fixture
def client():
    api.app.dependency_overrides[api.get_llm] = FakeLLM
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def _case(root, case_id, split):
    d = root / case_id
    d.mkdir(parents=True)
    (d / "referral_letter.txt").write_text(LETTER)
    (d / "labs.txt").write_text(LABS)
    (d / "gold.json").write_text(json.dumps({
        "case_id": case_id, "checklist_id": "cardiology-v1", "reference_date": "2026-11-01", "split": split,
        "items": [{"item_id": "reason", "status": "present"}, {"item_id": "bmp", "status": "present"}]}))
    return d


@pytest.fixture
def samples(tmp_path, monkeypatch):
    root = tmp_path / "cases"
    _case(root, "demo-dev-001", "dev")
    _case(root, "demo-test-001", "test")
    monkeypatch.setattr(api, "SAMPLE_DIRS", [root, tmp_path / "missing"])
    api._samples.cache_clear()
    yield root
    api._samples.cache_clear()


def test_health_and_checklists(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "cardiology-v1" in client.get("/checklists").json()


def test_samples_never_list_or_run_the_test_split(client, samples):
    listed = client.get("/samples").json()
    assert [s["case_id"] for s in listed] == ["demo-dev-001"]
    assert listed[0] == {"case_id": "demo-dev-001", "split": "dev", "referral_date": "2026-11-01",
                         "documents": 2, "pages": 2}
    assert client.post("/samples/demo-test-001/check").status_code == 404
    assert client.post("/samples/..%2Fcases%2Fdemo-dev-001/check").status_code == 404


def test_check_sample_returns_statuses_citations_gold_and_drafts(client, samples):
    r = client.post("/samples/demo-dev-001/check")
    assert r.status_code == 200
    body = r.json()
    items = {i["item_id"]: i for i in body["result"]["items"]}
    assert set(items) == {i.item_id for i in api.load_checklist("cardiology-v1").items}
    assert items["bmp"]["status"] == "present" and items["bmp"]["citations"][0]["doc_id"] == "labs"
    assert body["gold"] == {"reason": "present", "bmp": "present"}
    assert {d["doc_id"] for d in body["documents"]} == {"labs", "referral_letter"}
    missing = {i for i, it in items.items() if it["status"] == "missing"}
    assert missing and missing <= {f["item_id"] for f in body["result"]["followups"]}
    assert body["llm_calls"] > 0 and body["seconds"] >= 0


def test_upload_runs_the_pipeline_and_validates_input(client):
    files = [("files", ("referral_letter.txt", LETTER, "text/plain")), ("files", ("labs.txt", LABS, "text/plain"))]
    form = {"checklist_id": "cardiology-v1", "referral_date": "2026-11-01"}
    r = client.post("/cases", data=form, files=files)
    assert r.status_code == 200 and r.json()["gold"] is None
    assert {i["item_id"]: i["status"] for i in r.json()["result"]["items"]}["bmp"] == "present"

    bad_list = client.post("/cases", data={**form, "checklist_id": "../checklists/cardiology-v1"}, files=files)
    assert bad_list.status_code == 404
    exe = client.post("/cases", data=form, files=[("files", ("run.sh", "echo hi", "text/plain"))])
    assert exe.status_code == 400


def test_review_page_and_checklist_endpoint(client):
    page = client.get("/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    for hook in ('id="sample-form"', 'id="upload-form"', 'id="items"', 'id="followups"', "/samples/"):
        assert hook in page.text
    cl = client.get("/checklists/cardiology-v1").json()
    assert {i["item_id"] for i in cl["items"]} >= {"meds", "inr"}
    assert next(i for i in cl["items"] if i["item_id"] == "meds")["definition"]
    assert client.get("/checklists/nope").status_code == 404
