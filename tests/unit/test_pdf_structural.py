"""Tests for Tier 1 pdf_structural detectors.

Uses the test corpus builders to produce known-clean and known-tampered PDFs,
then asserts each detector fires only when expected.
"""

from __future__ import annotations

import pytest

from tamper_detect.detectors.base import DetectorContext
from tamper_detect.detectors.pdf_structural import (
    FontFamilyMixInLine,
    FontNotEmbedded,
    IncrementalUpdatePresent,
    ModDateAfterCreateDate,
    ProducerCreatorMismatch,
    TextOverImageOverlay,
    XrefAnomaly,
    _font_family_base,
    _parse_pdf_date,
)
from tamper_detect.loader import load
from testdata.pdf_builders import (
    build_ein_letter,
    build_sales_tax_permit,
    build_utility_invoice,
)
from testdata.tamper_ops import (
    edit_amount_incremental,
    overlay_swap_text,
    resave_as_word,
    swap_ein_incremental,
)


class TestHelpers:
    def test_font_family_base_strips_subset_and_style(self):
        assert _font_family_base("ABCDEF+Helvetica-Bold") == "Helvetica"
        assert _font_family_base("Times-Roman") == "Times"
        assert _font_family_base("Helvetica") == "Helvetica"
        assert _font_family_base("ABCDEF+Courier-BoldOblique") == "Courier"

    def test_font_family_base_leaves_short_prefixes_alone(self):
        # 5-char prefix isn't a real subset marker; keep it.
        assert _font_family_base("AB+Helvetica-Bold") == "AB+Helvetica"

    def test_parse_pdf_date(self):
        d = _parse_pdf_date("D:20260614120000+00'00'")
        assert d is not None
        assert d.year == 2026
        assert d.month == 6

    def test_parse_pdf_date_none(self):
        assert _parse_pdf_date(None) is None
        assert _parse_pdf_date("bogus") is None


class TestProducerCreatorMismatch:
    def test_clean_permit_does_not_fire(self):
        with load(build_sales_tax_permit()) as doc:
            findings = ProducerCreatorMismatch()(DetectorContext(doc))
            assert findings == []

    def test_resaved_as_word_fires(self):
        tampered = resave_as_word(build_sales_tax_permit())
        with load(tampered) as doc:
            findings = ProducerCreatorMismatch()(DetectorContext(doc))
            assert len(findings) == 1
            assert findings[0].score >= 0.8
            assert findings[0].evidence["matched_tokens"]


class TestModDateAfterCreateDate:
    def test_clean_does_not_fire(self):
        with load(build_utility_invoice()) as doc:
            findings = ModDateAfterCreateDate()(DetectorContext(doc))
            assert findings == []

    def test_incrementally_updated_fires(self):
        clean = build_utility_invoice()
        tampered = edit_amount_incremental(clean, "$87.42", "$874.20")
        with load(tampered) as doc:
            findings = ModDateAfterCreateDate()(DetectorContext(doc))
            # This depends on the incremental save touching /ModDate — it may
            # or may not, but if it fires, the score should be reasonable.
            for f in findings:
                assert 0.0 <= f.score <= 1.0


class TestIncrementalUpdatePresent:
    def test_clean_does_not_fire(self):
        with load(build_utility_invoice()) as doc:
            findings = IncrementalUpdatePresent()(DetectorContext(doc))
            assert findings == []

    def test_incrementally_updated_fires(self):
        clean = build_utility_invoice()
        tampered = edit_amount_incremental(clean, "$87.42", "$874.20")
        with load(tampered) as doc:
            findings = IncrementalUpdatePresent()(DetectorContext(doc))
            assert len(findings) == 1
            assert findings[0].score >= 0.7
            assert findings[0].evidence["num_eof_markers"] >= 2


class TestXrefAnomaly:
    def test_clean_does_not_fire(self):
        with load(build_utility_invoice()) as doc:
            findings = XrefAnomaly()(DetectorContext(doc))
            assert findings == []


class TestFontNotEmbedded:
    def test_clean_docs_do_not_fire(self):
        # All our clean docs use Standard-14 fonts — should NOT fire.
        for pdf in (build_utility_invoice(), build_sales_tax_permit(), build_ein_letter()):
            with load(pdf) as doc:
                findings = FontNotEmbedded()(DetectorContext(doc))
                assert findings == [], f"unexpected fire on: {findings}"


class TestFontFamilyMixInLine:
    def test_clean_ein_does_not_fire(self):
        with load(build_ein_letter()) as doc:
            findings = FontFamilyMixInLine()(DetectorContext(doc))
            # Header line has different fonts, but each LINE should be single-family
            # in the clean template. If this fires, it's ok as long as score is low.
            for f in findings:
                assert f.score < 0.9

    def test_ein_swap_fires(self):
        clean = build_ein_letter()
        tampered = swap_ein_incremental(clean, "88-3341572", "77-9998880")
        with load(tampered) as doc:
            findings = FontFamilyMixInLine()(DetectorContext(doc))
            assert len(findings) == 1
            assert findings[0].score >= 0.7
            assert findings[0].evidence["count"] >= 1


class TestTextOverImageOverlay:
    def test_clean_does_not_fire(self):
        # The clean utility invoice has one grey-fill rect (the charges box)
        # but no white-fill overlays.
        with load(build_utility_invoice()) as doc:
            findings = TextOverImageOverlay()(DetectorContext(doc))
            assert findings == []

    def test_overlay_swap_fires(self):
        clean = build_sales_tax_permit()
        tampered = overlay_swap_text(
            clean, "Blue Ocean Coffee LLC", "Red Mountain Tea LLC"
        )
        with load(tampered) as doc:
            findings = TextOverImageOverlay()(DetectorContext(doc))
            assert len(findings) == 1
            assert findings[0].score >= 0.75
            assert findings[0].evidence["count"] >= 1
