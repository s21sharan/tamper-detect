"""Tests for the Finding / Report data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tamper_detect.report import DocType, Finding, Report, ReportMeta


class TestFinding:
    def test_minimal_finding(self):
        f = Finding(signal="jpeg_ghost", tier="pixel", score=0.7)
        assert f.signal == "jpeg_ghost"
        assert f.tier == "pixel"
        assert f.score == 0.7
        assert f.evidence == {}
        assert f.weight_applied is None

    def test_evidence_is_arbitrary(self):
        f = Finding(
            signal="ela_hotspots",
            tier="pixel",
            score=0.5,
            evidence={"page": 1, "bbox": [0, 0, 100, 100], "note": "hot"},
        )
        assert f.evidence["bbox"] == [0, 0, 100, 100]

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Finding(signal="x", tier="pixel", score=1.5)
        with pytest.raises(ValidationError):
            Finding(signal="x", tier="pixel", score=-0.1)

    def test_bad_tier_rejected(self):
        with pytest.raises(ValidationError):
            Finding(signal="x", tier="bogus", score=0.5)

    def test_empty_signal_rejected(self):
        with pytest.raises(ValidationError):
            Finding(signal="   ", tier="pixel", score=0.5)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            Finding(signal="x", tier="pixel", score=0.5, unknown_field=1)


class TestReport:
    def test_minimal_report(self):
        r = Report(
            document_id="sha256:aaa",
            overall_score=0.0,
            decision="pass",
        )
        assert r.doc_type_hint == DocType.UNKNOWN
        assert r.findings == []
        assert r.narrative is None
        assert r.meta.version == "0.1.0"

    def test_full_report(self):
        r = Report(
            document_id="sha256:bbb",
            doc_type_hint=DocType.UTILITY_INVOICE,
            overall_score=0.72,
            decision="fail",
            findings=[
                Finding(
                    signal="font_family_mix_in_line",
                    tier="structural",
                    score=0.9,
                    weight_applied=0.9,
                    evidence={"page": 1},
                )
            ],
            narrative="Two signs of editing detected.",
            meta=ReportMeta(detectors_run=["pdf_structural"], runtime_ms=180),
        )
        assert r.decision == "fail"
        assert r.findings[0].weight_applied == 0.9
        assert r.meta.runtime_ms == 180

    def test_overall_score_range(self):
        with pytest.raises(ValidationError):
            Report(document_id="x", overall_score=1.5, decision="pass")

    def test_bad_decision_rejected(self):
        with pytest.raises(ValidationError):
            Report(document_id="x", overall_score=0.1, decision="maybe")

    def test_json_roundtrip(self):
        r = Report(
            document_id="sha256:ccc",
            overall_score=0.3,
            decision="review",
            findings=[Finding(signal="ela_hotspots", tier="pixel", score=0.3)],
        )
        raw = r.model_dump_json()
        r2 = Report.model_validate_json(raw)
        assert r2 == r
