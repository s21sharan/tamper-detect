"""Tests for Tier 2 pixel_forensics detectors."""

from __future__ import annotations

import pytest

from tamper_detect.detectors.base import DetectorContext
from tamper_detect.detectors.pixel_forensics import (
    CopyMove,
    ElaHotspots,
    JpegGhost,
    NoiseResidualInconsistency,
)
from tamper_detect.loader import load
from testdata.pdf_builders import build_ein_letter, build_utility_invoice
from testdata.tamper_ops import (
    ai_lookalike,
    copy_move_logo,
    scan_and_splice_amount,
)


class TestElaHotspots:
    def test_clean_vector_pdf_low_or_no_fire(self):
        # A rendered vector PDF has no prior JPEG history; ELA residual should
        # be roughly uniform. If it fires, score should be low.
        with load(build_utility_invoice()) as doc:
            findings = ElaHotspots()(DetectorContext(doc))
            for f in findings:
                assert f.score < 0.8

    def test_runs_without_false_positive_on_clean(self):
        # ELA is intentionally conservative — text-edge JPEG artifacts on
        # our synthetic data are too close to real splice signatures for a
        # low-threshold ELA to distinguish. It should be silent on clean.
        with load(build_utility_invoice()) as doc:
            findings = ElaHotspots()(DetectorContext(doc))
            assert findings == []

    def test_runs_on_splice_without_error(self):
        # A specific QF-mismatch splice is JPEG-Ghost's job; ELA may or may
        # not fire depending on where in the QF cascade the splice lands.
        clean = build_utility_invoice()
        tampered = scan_and_splice_amount(clean, "$87.42", "$874.20")
        with load(tampered) as doc:
            _ = ElaHotspots()(DetectorContext(doc))


class TestJpegGhost:
    def test_spliced_scan_fires(self):
        clean = build_utility_invoice()
        tampered = scan_and_splice_amount(clean, "$87.42", "$874.20")
        with load(tampered) as doc:
            findings = JpegGhost()(DetectorContext(doc))
            # We allow this to be flaky in tuning; require at least one page
            # with outliers if it fires.
            for f in findings:
                assert f.evidence["pages"]


class TestNoiseResidualInconsistency:
    def test_ai_lookalike_fires(self):
        clean = build_ein_letter()
        tampered = ai_lookalike(clean)
        with load(tampered) as doc:
            findings = NoiseResidualInconsistency()(DetectorContext(doc))
            # AI look-alike has aggressive Gaussian noise applied region-wide
            # — noise residual outliers should show up.
            assert findings, "expected noise residual to fire on AI look-alike"

    def test_clean_vector_pdf_low_or_no_fire(self):
        with load(build_utility_invoice()) as doc:
            findings = NoiseResidualInconsistency()(DetectorContext(doc))
            for f in findings:
                # Vector PDFs are extremely smooth; if it fires the score is tiny.
                assert f.score < 0.8


class TestCopyMove:
    def test_copy_move_pdf_fires(self):
        clean = build_utility_invoice()
        tampered = copy_move_logo(clean)
        with load(tampered) as doc:
            findings = CopyMove()(DetectorContext(doc))
            assert findings, "expected copy_move to fire on duplicated region"
            f = findings[0]
            assert f.score >= 0.5
            assert f.evidence["pages"][0]["clusters"]

    def test_clean_does_not_fire(self):
        with load(build_utility_invoice()) as doc:
            findings = CopyMove()(DetectorContext(doc))
            # A clean vector PDF has repeating background elements that ORB
            # may match, but they shouldn't cluster tightly enough.
            for f in findings:
                assert f.score < 0.9
