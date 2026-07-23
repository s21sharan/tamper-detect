"""Tests for Tier 4 experimental detectors."""

from __future__ import annotations

import shutil

import pytest

from tamper_detect.detectors.ai_generation import AiGeneratedImage, AiGeneratedText
from tamper_detect.detectors.base import DetectorContext
from tamper_detect.detectors.template_match import TemplateMismatch
from tamper_detect.loader import load
from testdata.pdf_builders import (
    build_ein_letter,
    build_sales_tax_permit,
    build_utility_invoice,
)
from testdata.tamper_ops import ai_lookalike


class TestAiGeneratedImage:
    def test_clean_docs_do_not_fire(self):
        for pdf in (build_utility_invoice(), build_sales_tax_permit(), build_ein_letter()):
            with load(pdf) as doc:
                assert AiGeneratedImage()(DetectorContext(doc)) == []

    def test_ai_lookalike_fires_via_producer(self):
        clean = build_ein_letter()
        tampered = ai_lookalike(clean)
        with load(tampered) as doc:
            findings = AiGeneratedImage()(DetectorContext(doc))
            assert findings, "expected AI producer string to fire the detector"
            assert findings[0].score >= 0.6


class TestAiGeneratedText:
    def test_no_fire_on_normal_business_text(self):
        with load(build_utility_invoice()) as doc:
            assert AiGeneratedText()(DetectorContext(doc)) == []

    def test_fires_on_llm_leak_phrase(self):
        with load(build_utility_invoice()) as doc:
            ctx = DetectorContext(
                doc,
                supplied_ocr_text="As an AI language model, I cannot generate an invoice.",
            )
            findings = AiGeneratedText()(ctx)
            assert findings


class TestTemplateMismatch:
    @pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
    def test_ai_lookalike_ein_fires(self):
        # AI look-alike EIN is image-only, but the visible text (via OCR)
        # says it's an EIN letter. That structural shape is inconsistent for
        # a real IRS-issued document → fires.
        clean = build_ein_letter()
        tampered = ai_lookalike(clean)
        with load(tampered) as doc:
            findings = TemplateMismatch()(DetectorContext(doc))
            assert findings
            assert findings[0].evidence["doc_type"] == "ein_letter"

    def test_clean_ein_does_not_fire(self):
        # Clean EIN letter is vector-text — matches the expected template shape.
        with load(build_ein_letter()) as doc:
            findings = TemplateMismatch()(DetectorContext(doc))
            assert findings == []

    def test_clean_utility_invoice_does_not_fire(self):
        with load(build_utility_invoice()) as doc:
            findings = TemplateMismatch()(DetectorContext(doc))
            assert findings == []
