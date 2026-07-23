"""Tests for the OCR-vs-render diff detector.

These call real tesseract via pytesseract. If tesseract is not available on
the system, tests skip gracefully.
"""

from __future__ import annotations

import shutil

import pytest

from tamper_detect.detectors.base import DetectorContext
from tamper_detect.detectors.ocr_render_diff import OcrRenderDiff, _significant, _tokens
from tamper_detect.loader import load
from testdata.pdf_builders import (
    build_ein_letter,
    build_sales_tax_permit,
    build_utility_invoice,
)
from testdata.tamper_ops import (
    edit_amount_incremental,
    overlay_swap_text,
    scan_and_splice_amount,
    swap_ein_incremental,
)


pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="tesseract not installed",
)


class TestTokenHelpers:
    def test_tokens_normalizes(self):
        toks = _tokens("Hello, World! $87.42")
        assert "hello" in toks
        assert "world" in toks
        assert "$87.42" in toks

    def test_significant_matches_currency(self):
        assert _significant("$87.42")
        assert _significant("874.20")
        assert _significant("88-3341572")
        assert _significant("stp-2026-118-4471")
        assert not _significant("hello")


class TestOcrRenderDiff:
    def test_clean_utility_invoice_does_not_fire(self):
        with load(build_utility_invoice()) as doc:
            findings = OcrRenderDiff()(DetectorContext(doc))
            # A clean rendered PDF should closely match its own text layer.
            # If it fires (OCR noise), the score should be low.
            for f in findings:
                assert f.score < 0.8

    def test_amount_edit_incremental_fires(self):
        # Text layer says $87.42 (old + new). Render shows the new amount
        # on top of a white rect.
        clean = build_utility_invoice()
        tampered = edit_amount_incremental(clean, "$87.42", "$874.20")
        with load(tampered) as doc:
            findings = OcrRenderDiff()(DetectorContext(doc))
            assert findings, "expected diff to fire on amount edit"
            f = findings[0]
            assert f.score >= 0.5

    def test_scanned_splice_fires(self):
        clean = build_utility_invoice()
        tampered = scan_and_splice_amount(clean, "$87.42", "$874.20")
        with load(tampered) as doc:
            findings = OcrRenderDiff()(DetectorContext(doc))
            assert findings, "expected diff to fire on scanned splice"

    def test_business_name_overlay_runs_without_error(self):
        # When old and new names share tokens with other unchanged text in
        # the doc (e.g. "Blue Ocean Coffee" also appears in the DBA field),
        # the token-set diff can't isolate the swap. That case is caught by
        # text_over_image_overlay instead.
        clean = build_sales_tax_permit()
        tampered = overlay_swap_text(clean, "Blue Ocean Coffee LLC", "Red Mountain Tea LLC")
        with load(tampered) as doc:
            _ = OcrRenderDiff()(DetectorContext(doc))

    def test_supplied_ocr_text_is_used(self):
        # If the caller passes a reference OCR string that matches the render,
        # the detector should not fire even if the text layer contains extra
        # (hidden) content.
        clean = build_utility_invoice()
        tampered = edit_amount_incremental(clean, "$87.42", "$874.20")
        with load(tampered) as doc:
            # Supplied OCR reflects what's visible on the render.
            ctx = DetectorContext(doc=doc, supplied_ocr_text="$874.20 Metro Water")
            findings = OcrRenderDiff()(ctx)
            # We can't guarantee zero findings because the render OCR sees many
            # more tokens than "$874.20 Metro Water", but if a finding does fire
            # it should NOT list $874.20 as missing_from_ocr.
            for f in findings:
                assert "$874.20" not in f.evidence.get("significant_missing_from_ocr", [])

    def test_ein_swap_fires(self):
        clean = build_ein_letter()
        tampered = swap_ein_incremental(clean, "88-3341572", "77-9998880")
        with load(tampered) as doc:
            findings = OcrRenderDiff()(DetectorContext(doc))
            assert findings, "expected diff to fire on EIN swap"
            f = findings[0]
            # Old EIN should appear in text layer but be missing from OCR
            # (since the overlay covers it). The new EIN should be in OCR
            # but the text layer contains BOTH (old + new), so the "missing
            # from OCR" side is the informative one.
            sig_missing = f.evidence.get("significant_missing_from_ocr", [])
            assert any("88-3341572" in t or "3341572" in t for t in sig_missing), (
                f"expected old EIN in missing_from_ocr, got: {sig_missing}"
            )
