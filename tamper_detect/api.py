"""FastAPI surface — POST /analyze, POST /analyze-batch, GET /healthz.

Run with:  uvicorn tamper_detect.api:app
(Not launched during tests or by tools that don't own a server process.)
"""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from tamper_detect.analyze import analyze
from tamper_detect.report import Report


app = FastAPI(title="tamper-detect", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=Report)
async def analyze_endpoint(
    pdf: UploadFile = File(..., description="The PDF to analyze"),
    ocr_text: str | None = Form(None, description="Optional reference OCR text from upstream pipeline"),
) -> Report:
    pdf_bytes = await pdf.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="empty PDF upload")
    try:
        return analyze(pdf_bytes, supplied_ocr_text=ocr_text)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"analyze failed: {exc}") from exc


@app.post("/analyze-batch", response_model=list[Report])
async def analyze_batch(
    pdfs: list[UploadFile] = File(...),
) -> list[Report]:
    reports: list[Report] = []
    for pdf in pdfs:
        data = await pdf.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"empty PDF: {pdf.filename}")
        reports.append(analyze(data))
    return reports
