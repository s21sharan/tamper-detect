"""Tests for the top-level analyze() entry point."""

from __future__ import annotations

from tamper_detect.analyze import analyze
from tamper_detect.report import DocType
from testdata.pdf_builders import build_ein_letter, build_utility_invoice
from testdata.tamper_ops import edit_amount_incremental, overlay_swap_text


class TestAnalyze:
    def test_clean_utility_invoice_passes(self):
        report = analyze(build_utility_invoice(), enable_narrative=False)
        assert report.decision == "pass"
        assert report.overall_score < 0.3
        assert report.doc_type_hint in (DocType.UTILITY_INVOICE, DocType.UNKNOWN)
        assert report.meta.detectors_run

    def test_clean_ein_passes(self):
        report = analyze(build_ein_letter(), enable_narrative=False)
        assert report.decision == "pass"

    def test_amount_edit_flagged(self):
        clean = build_utility_invoice()
        tampered = edit_amount_incremental(clean, "$87.42", "$874.20")
        report = analyze(tampered, enable_narrative=False)
        assert report.decision != "pass"
        signals = {f.signal for f in report.findings}
        assert "incremental_update_present" in signals

    def test_overlay_swap_flagged(self):
        clean = build_utility_invoice()
        tampered = overlay_swap_text(clean, "Blue Ocean Coffee LLC", "Blue Ocean Coffee LLD")
        # tiny change but text is still overlaid
        report = analyze(tampered, enable_narrative=False)
        assert report.decision != "pass" or any(
            f.signal == "text_over_image_overlay" for f in report.findings
        )
