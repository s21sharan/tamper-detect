"""Tests for the detector protocol, registry, and runner."""

from __future__ import annotations

import io

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from tamper_detect.detectors.base import (
    BaseDetector,
    DetectorContext,
    clear_registry,
    register,
    registry,
    run_detectors,
)
from tamper_detect.loader import load
from tamper_detect.report import Finding


def _tiny_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.drawString(72, 720, "hi")
    c.showPage()
    c.save()
    return buf.getvalue()


class _FakeDetector(BaseDetector):
    def __init__(self, name: str, findings: list[Finding]) -> None:
        self.name = name
        self._findings = findings
        self.call_count = 0

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        self.call_count += 1
        return list(self._findings)


class _RaisingDetector(BaseDetector):
    name = "raiser"

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        raise RuntimeError("boom")


class TestRegistry:
    def setup_method(self):
        clear_registry()

    def test_register_and_snapshot(self):
        d = _FakeDetector("alpha", [])
        register(d)
        assert "alpha" in registry()
        assert registry()["alpha"] is d

    def test_register_is_idempotent_by_name(self):
        d1 = _FakeDetector("dup", [])
        d2 = _FakeDetector("dup", [])
        register(d1)
        register(d2)
        assert registry()["dup"] is d2

    def test_snapshot_is_a_copy(self):
        d = _FakeDetector("alpha", [])
        register(d)
        snap = registry()
        snap["injected"] = _FakeDetector("injected", [])
        assert "injected" not in registry()


class TestRunDetectors:
    def setup_method(self):
        clear_registry()

    def test_run_returns_findings_and_names(self):
        f1 = Finding(signal="a", tier="structural", score=0.2)
        f2 = Finding(signal="b", tier="pixel", score=0.4)
        d1 = _FakeDetector("d1", [f1])
        d2 = _FakeDetector("d2", [f2])
        with load(_tiny_pdf()) as doc:
            ctx = DetectorContext(doc=doc)
            findings, names = run_detectors(ctx, [d1, d2])
        assert [f.signal for f in findings] == ["a", "b"]
        assert names == ["d1", "d2"]
        assert d1.call_count == 1
        assert d2.call_count == 1

    def test_run_uses_registry_when_no_list(self):
        f = Finding(signal="a", tier="structural", score=0.5)
        register(_FakeDetector("only", [f]))
        with load(_tiny_pdf()) as doc:
            ctx = DetectorContext(doc=doc)
            findings, names = run_detectors(ctx)
        assert names == ["only"]
        assert findings[0].signal == "a"

    def test_raising_detector_is_skipped(self):
        good = _FakeDetector("good", [Finding(signal="ok", tier="structural", score=0.1)])
        bad = _RaisingDetector()
        with load(_tiny_pdf()) as doc:
            ctx = DetectorContext(doc=doc)
            findings, names = run_detectors(ctx, [bad, good])
        assert "raiser" not in names
        assert "good" in names
        assert findings[0].signal == "ok"
