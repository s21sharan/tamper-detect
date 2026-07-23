"""Tests for the scorer + weights loading + decision thresholds."""

from __future__ import annotations

import pytest
import yaml

from tamper_detect.report import Finding
from tamper_detect.scorer import (
    Weights,
    apply_weights,
    decide,
    load_default_weights,
    score,
    score_and_decide,
)


def _weights(
    tiers: dict[str, float] | None = None,
    signals: dict[str, float] | None = None,
    thresholds: dict[str, tuple[float, float]] | None = None,
) -> Weights:
    return Weights.from_dict(
        {
            "tiers": tiers or {"structural": 1.0, "pixel": 1.0, "ocr_diff": 1.0, "experimental": 1.0},
            "signals": signals or {},
            "decision_thresholds": {
                "pass": list(thresholds.get("pass", (0.0, 0.3))) if thresholds else [0.0, 0.3],
                "review": list(thresholds.get("review", (0.3, 0.65))) if thresholds else [0.3, 0.65],
                "fail": list(thresholds.get("fail", (0.65, 1.01))) if thresholds else [0.65, 1.01],
            },
        }
    )


class TestWeights:
    def test_from_dict_roundtrip(self):
        w = _weights(signals={"a": 0.5, "b": 1.0})
        assert w.signals["a"] == 0.5
        assert w.tiers["structural"] == 1.0
        assert w.thresholds["pass"] == (0.0, 0.3)

    def test_missing_threshold_rejected(self):
        with pytest.raises(ValueError):
            Weights.from_dict(
                {
                    "tiers": {},
                    "signals": {},
                    "decision_thresholds": {"pass": [0, 0.3], "review": [0.3, 0.65]},
                }
            )

    def test_malformed_threshold_rejected(self):
        with pytest.raises(ValueError):
            Weights.from_dict(
                {
                    "tiers": {},
                    "signals": {},
                    "decision_thresholds": {"pass": [0.0], "review": [0.3, 0.65], "fail": [0.65, 1.01]},
                }
            )

    def test_load_default_weights(self):
        w = load_default_weights()
        assert "structural" in w.tiers
        assert "jpeg_ghost" in w.signals
        assert w.thresholds["pass"][0] == 0.0

    def test_load_from_yaml_file(self, tmp_path):
        yml = tmp_path / "w.yaml"
        yml.write_text(
            yaml.safe_dump(
                {
                    "tiers": {"structural": 1.0, "pixel": 0.5, "ocr_diff": 1.0, "experimental": 0.1},
                    "signals": {"foo": 0.7},
                    "decision_thresholds": {"pass": [0, 0.4], "review": [0.4, 0.7], "fail": [0.7, 1.01]},
                }
            )
        )
        w = Weights.load(yml)
        assert w.signals["foo"] == 0.7
        assert w.tiers["experimental"] == 0.1


class TestScoring:
    def test_empty_findings_score_zero(self):
        w = _weights()
        assert score([], w) == 0.0

    def test_apply_weights_mutates_finding(self):
        w = _weights(
            tiers={"pixel": 0.8, "structural": 1.0, "ocr_diff": 1.0, "experimental": 1.0},
            signals={"jpeg_ghost": 0.85},
        )
        f = Finding(signal="jpeg_ghost", tier="pixel", score=0.5)
        contrib = apply_weights(f, w)
        assert f.weight_applied == pytest.approx(0.8 * 0.85)
        assert contrib == pytest.approx(0.8 * 0.85 * 0.5)

    def test_unknown_signal_defaults_to_signal_weight_one(self):
        w = _weights(tiers={"structural": 1.0, "pixel": 1.0, "ocr_diff": 1.0, "experimental": 1.0})
        f = Finding(signal="brand_new_signal", tier="structural", score=0.5)
        contrib = apply_weights(f, w)
        assert contrib == pytest.approx(0.5)  # 1 * 1 * 0.5

    def test_score_sums_and_clamps(self):
        w = _weights(
            signals={"a": 1.0, "b": 1.0, "c": 1.0},
        )
        findings = [
            Finding(signal="a", tier="structural", score=0.6),
            Finding(signal="b", tier="pixel", score=0.6),
            Finding(signal="c", tier="ocr_diff", score=0.6),
        ]
        s = score(findings, w)
        # Uncapped would be 1.8; clamp to 1.0.
        assert s == 1.0

    def test_score_below_cap(self):
        w = _weights(
            tiers={"structural": 0.5, "pixel": 0.5, "ocr_diff": 0.5, "experimental": 0.5},
            signals={"a": 0.4},
        )
        f = Finding(signal="a", tier="structural", score=0.5)
        # 0.5 * 0.4 * 0.5 = 0.10
        assert score([f], w) == pytest.approx(0.10)


class TestDecision:
    def test_pass_boundary(self):
        w = _weights()
        assert decide(0.0, w) == "pass"
        assert decide(0.29, w) == "pass"

    def test_review_band(self):
        w = _weights()
        assert decide(0.30, w) == "review"
        assert decide(0.64, w) == "review"

    def test_fail_band(self):
        w = _weights()
        assert decide(0.65, w) == "fail"
        assert decide(1.0, w) == "fail"


class TestScoreAndDecide:
    def test_end_to_end(self):
        w = _weights(
            tiers={"structural": 1.0, "pixel": 1.0, "ocr_diff": 1.0, "experimental": 1.0},
            signals={"a": 1.0},
        )
        s, d = score_and_decide(
            [Finding(signal="a", tier="structural", score=0.9)],
            w,
        )
        assert s == pytest.approx(0.9)
        assert d == "fail"

    def test_uses_default_weights_when_none(self):
        # Just make sure it doesn't error and produces a plausible decision.
        s, d = score_and_decide(
            [Finding(signal="jpeg_ghost", tier="pixel", score=1.0)]
        )
        assert 0.0 <= s <= 1.0
        assert d in {"pass", "review", "fail"}
