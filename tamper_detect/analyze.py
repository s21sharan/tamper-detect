"""High-level analyze() — the entry point every surface (CLI, FastAPI, tests)
calls. Runs all registered detectors, applies the scorer, and (optionally)
attaches a Claude-generated narrative.
"""

from __future__ import annotations

import os
import time

# Importing these modules registers all detectors as a side effect.
from tamper_detect.detectors import base, pdf_structural, pixel_forensics, ocr_render_diff, ai_generation, template_match  # noqa: F401
from tamper_detect.detectors.base import DetectorContext, run_detectors
from tamper_detect.loader import LoadedDocument
from tamper_detect.report import DocType, Report, ReportMeta
from tamper_detect.scorer import Weights, load_default_weights, score_and_decide


def analyze(
    pdf_bytes: bytes,
    supplied_ocr_text: str | None = None,
    weights: Weights | None = None,
    enable_narrative: bool | None = None,
) -> Report:
    """Analyze a single PDF. Returns a Report.

    `enable_narrative` overrides the ENABLE_NARRATIVE env var when set.
    """
    t0 = time.perf_counter()
    with LoadedDocument(pdf_bytes) as doc:
        ctx = DetectorContext(doc=doc, supplied_ocr_text=supplied_ocr_text)
        findings, detectors_run = run_detectors(ctx)
        w = weights or load_default_weights()
        overall, decision = score_and_decide(findings, w)

        # Simple doc-type hint from the template_match signatures if visible
        # in the text layer or supplied text — cheap, informative.
        doc_type_hint = _guess_doc_type(supplied_ocr_text or doc.all_text())

        report = Report(
            document_id=doc.document_id,
            doc_type_hint=doc_type_hint,
            overall_score=overall,
            decision=decision,
            findings=findings,
            meta=ReportMeta(
                detectors_run=detectors_run,
                runtime_ms=int((time.perf_counter() - t0) * 1000),
            ),
        )

    _flag = enable_narrative if enable_narrative is not None else os.getenv("ENABLE_NARRATIVE") == "1"
    if _flag and report.findings:
        try:
            # Lazy import so tests that don't touch narrative don't need the SDK.
            from tamper_detect.narrative import generate_narrative

            report.narrative = generate_narrative(report)
        except Exception as e:  # pragma: no cover
            report.narrative = f"[narrative unavailable: {e}]"

    return report


def _guess_doc_type(text: str) -> DocType:
    from tamper_detect.detectors.template_match import DOC_SIGNATURES

    t = (text or "").lower()
    best_hits = 0
    best: str | None = None
    for name, sigs in DOC_SIGNATURES.items():
        h = sum(1 for s in sigs if s in t)
        if h > best_hits:
            best_hits = h
            best = name
    if best_hits < 2 or best is None:
        return DocType.UNKNOWN
    return DocType(best)
