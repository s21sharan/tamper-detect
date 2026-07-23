"""Tier 4 — template consistency check.

Classifies documents by content signatures (utility invoice / sales-tax permit
/ IRS EIN letter). Fires `template_mismatch` when a document is classifiable
but its structural shape doesn't match what that issuer would normally
produce — most notably, an image-only PDF for a doc type that is normally
issued as vector-text PDF.
"""

from __future__ import annotations

import pytesseract

from tamper_detect.detectors.base import BaseDetector, DetectorContext, register
from tamper_detect.report import Finding


# doc_type -> list of lowercase signature substrings; a document is classified
# to a type if at least MIN_SIGNATURE_HITS signatures appear in its visible
# text (text layer or OCR of render).
DOC_SIGNATURES = {
    "utility_invoice": [
        "utility",
        "invoice",
        "account number",
        "amount due",
        "total due",
        "billing",
        "kwh",
        "gallon",
    ],
    "sales_tax_permit": [
        "sales tax permit",
        "department of revenue",
        "permit number",
        "state of",
        "authorizes the holder",
    ],
    "ein_letter": [
        "employer identification number",
        "internal revenue service",
        "department of the treasury",
        "ein",
        "irs",
    ],
}

MIN_SIGNATURE_HITS = 2

# Doc types that are typically issued as vector-text PDFs from an online
# system. An image-only representation of these is suspicious — a real
# customer would send the source PDF, not a phone photo of one.
VECTOR_ONLY_TYPES = {"ein_letter"}


class TemplateMismatch(BaseDetector):
    name = "template_mismatch"

    _OCR_DPI = 200

    def _classify(self, text: str) -> tuple[str | None, dict[str, int]]:
        text = text.lower()
        hits: dict[str, int] = {}
        for dtype, sigs in DOC_SIGNATURES.items():
            hits[dtype] = sum(1 for s in sigs if s in text)
        # Pick the type with the most hits above threshold.
        best = max(hits.items(), key=lambda kv: kv[1])
        if best[1] >= MIN_SIGNATURE_HITS:
            return best[0], hits
        return None, hits

    def _get_visible_text(self, ctx: DetectorContext) -> tuple[str, str]:
        """Return (visible_text, source) — supplied, text_layer, or ocr."""
        if ctx.supplied_ocr_text:
            return ctx.supplied_ocr_text, "supplied"
        layer = ctx.doc.all_text().strip()
        if layer:
            return layer, "text_layer"
        # Fall back to OCR of the rendered first page.
        try:
            img = ctx.doc.render_page(0, dpi=self._OCR_DPI)
            return pytesseract.image_to_string(img), "ocr_render"
        except Exception:
            return "", "none"

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        text, source = self._get_visible_text(ctx)
        if not text.strip():
            return []
        doc_type, hits = self._classify(text)
        if doc_type is None:
            return []

        # Structural check: is the document image-only?
        is_image_only = not ctx.doc.all_text().strip()

        if is_image_only and doc_type in VECTOR_ONLY_TYPES:
            return [
                Finding(
                    signal=self.name,
                    tier="experimental",
                    score=0.75,
                    evidence={
                        "doc_type": doc_type,
                        "classified_from": source,
                        "signature_hits": hits,
                        "reason": (
                            f"{doc_type} normally issued as vector-text PDF; "
                            "received image-only PDF"
                        ),
                    },
                )
            ]
        return []


register(TemplateMismatch())
