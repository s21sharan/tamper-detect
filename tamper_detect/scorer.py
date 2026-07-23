"""Scorer — combine detector findings into a single overall_score + decision.

Formula: overall_score = min(1.0, sum over findings of
             tier_weight[finding.tier] * signal_weight[finding.signal] * finding.score)

An unknown signal contributes at signal_weight=1.0 (still gets the tier
weight). This makes it safe to add new detectors without immediately editing
weights.yaml, though production tuning should always name every signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tamper_detect.report import Decision, Finding, Tier


@dataclass(frozen=True)
class Weights:
    tiers: dict[str, float]
    signals: dict[str, float]
    thresholds: dict[str, tuple[float, float]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Weights":
        tiers = {str(k): float(v) for k, v in data.get("tiers", {}).items()}
        signals = {str(k): float(v) for k, v in data.get("signals", {}).items()}
        raw_thresh = data.get("decision_thresholds", {})
        thresholds: dict[str, tuple[float, float]] = {}
        for k, v in raw_thresh.items():
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError(f"decision_thresholds.{k} must be [low, high]")
            thresholds[str(k)] = (float(v[0]), float(v[1]))
        for required in ("pass", "review", "fail"):
            if required not in thresholds:
                raise ValueError(f"decision_thresholds.{required} is required")
        return cls(tiers=tiers, signals=signals, thresholds=thresholds)

    @classmethod
    def load(cls, path: str | Path) -> "Weights":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})


DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "weights.yaml"


def load_default_weights() -> Weights:
    return Weights.load(DEFAULT_WEIGHTS_PATH)


def apply_weights(finding: Finding, weights: Weights) -> float:
    """Return the (already-normalized) contribution of a finding: tier * signal * score.

    Also mutates finding.weight_applied for the report.
    """
    tier_w = weights.tiers.get(finding.tier, 1.0)
    signal_w = weights.signals.get(finding.signal, 1.0)
    combined = tier_w * signal_w
    finding.weight_applied = combined
    return combined * finding.score


def score(findings: list[Finding], weights: Weights) -> float:
    """Aggregate finding contributions into overall_score in [0, 1]."""
    total = sum(apply_weights(f, weights) for f in findings)
    return min(1.0, max(0.0, total))


def decide(overall_score: float, weights: Weights) -> Decision:
    """Map an overall_score to a Decision using the threshold bands."""
    # Iterate in preference order — a score that overlaps two bands
    # (shouldn't happen with well-formed thresholds) resolves to the first hit.
    for name in ("pass", "review", "fail"):
        low, high = weights.thresholds[name]
        if low <= overall_score < high:
            return name  # type: ignore[return-value]
    # Above the top of the "fail" band still counts as fail.
    return "fail"


def score_and_decide(
    findings: list[Finding],
    weights: Weights | None = None,
) -> tuple[float, Decision]:
    """One-shot: compute overall_score, decision. Mutates findings.weight_applied."""
    w = weights or load_default_weights()
    s = score(findings, w)
    return s, decide(s, w)


# Re-exported for convenience.
__all__ = [
    "Weights",
    "apply_weights",
    "score",
    "decide",
    "score_and_decide",
    "load_default_weights",
]

# Silence "unused import" — used in type hints of consumers.
_TIER: Tier | None = None  # noqa: F841 (typing marker)
