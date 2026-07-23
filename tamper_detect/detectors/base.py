"""Detector protocol + a small registry.

Every detector is a callable that takes a DetectorContext and returns
Findings. Detectors are expected to be pure, side-effect-free, and to fail
gracefully — a broken detector should return `[]` rather than raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from tamper_detect.loader import LoadedDocument
from tamper_detect.report import Finding


@dataclass
class DetectorContext:
    """Everything a detector may need to inspect a document."""

    doc: LoadedDocument
    # Optional pre-extracted OCR text supplied by the caller. Used by the
    # ocr_render_diff detector as a reference. `None` means "not supplied".
    supplied_ocr_text: str | None = None
    # Detector-specific options — reserved for future use.
    options: dict[str, object] = field(default_factory=dict)


class Detector(Protocol):
    """A detector produces zero or more Findings for a document."""

    name: str

    def __call__(self, ctx: DetectorContext) -> list[Finding]: ...


# ---------------------------------------------------------------------------
# Simple registry — detectors register themselves at import time.
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, Detector] = {}


def register(detector: Detector) -> Detector:
    """Register a detector by its `.name`. Idempotent — re-registering replaces."""
    _REGISTRY[detector.name] = detector
    return detector


def registry() -> dict[str, Detector]:
    """Return a snapshot of the current registry."""
    return dict(_REGISTRY)


def clear_registry() -> None:
    """For tests."""
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Runner — runs a set of detectors and collects findings.
# ---------------------------------------------------------------------------


def run_detectors(
    ctx: DetectorContext,
    detectors: list[Detector] | None = None,
) -> tuple[list[Finding], list[str]]:
    """Run each detector against the context. Returns (findings, detectors_run).

    A detector that raises is caught — its exception is swallowed and the
    detector is skipped so one bad detector cannot break the whole report.
    In a production system this would emit a metric; for a prototype we just
    continue.
    """
    if detectors is None:
        detectors = list(registry().values())
    findings: list[Finding] = []
    names_run: list[str] = []
    for det in detectors:
        try:
            out = det(ctx)
        except Exception:
            # Prototype: skip a broken detector rather than break the report.
            continue
        names_run.append(det.name)
        findings.extend(out)
    return findings, names_run


# ---------------------------------------------------------------------------
# Convenience base class for detectors that want inheritance instead of the
# bare Protocol. Concrete detectors can use either style.
# ---------------------------------------------------------------------------


class BaseDetector:
    name: str = ""

    def __call__(self, ctx: DetectorContext) -> list[Finding]:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Detector {self.name}>"
