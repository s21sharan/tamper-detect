"""Tests for the FastAPI surface — uses TestClient, no live server."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from tamper_detect.api import app
from testdata.pdf_builders import build_utility_invoice
from testdata.tamper_ops import edit_amount_incremental


client = TestClient(app)


class TestApi:
    def test_healthz(self):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_analyze_clean(self):
        pdf = build_utility_invoice()
        r = client.post(
            "/analyze",
            files={"pdf": ("bill.pdf", io.BytesIO(pdf), "application/pdf")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] == "pass"
        assert body["overall_score"] < 0.3

    def test_analyze_tampered(self):
        pdf = edit_amount_incremental(build_utility_invoice(), "$87.42", "$874.20")
        r = client.post(
            "/analyze",
            files={"pdf": ("bill.pdf", io.BytesIO(pdf), "application/pdf")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] != "pass"

    def test_analyze_empty_pdf_rejected(self):
        r = client.post(
            "/analyze",
            files={"pdf": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert r.status_code == 400

    def test_analyze_batch(self):
        clean = build_utility_invoice()
        tampered = edit_amount_incremental(clean, "$87.42", "$874.20")
        r = client.post(
            "/analyze-batch",
            files=[
                ("pdfs", ("clean.pdf", io.BytesIO(clean), "application/pdf")),
                ("pdfs", ("tampered.pdf", io.BytesIO(tampered), "application/pdf")),
            ],
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 2
        assert body[0]["decision"] == "pass"
        assert body[1]["decision"] != "pass"
