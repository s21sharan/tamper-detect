"""Tests for the narrative generator — mocked, no real API calls."""

from __future__ import annotations

import os

import pytest

from tamper_detect.narrative import generate_narrative
from tamper_detect.report import Finding, Report


def _tampered_report() -> Report:
    return Report(
        document_id="sha256:abcd",
        overall_score=0.72,
        decision="fail",
        findings=[
            Finding(signal="incremental_update_present", tier="structural", score=0.85, weight_applied=0.8),
            Finding(signal="text_layer_mismatches_render", tier="ocr_diff", score=0.9, weight_applied=0.81),
        ],
    )


def test_raises_without_credentials(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no AI credentials"):
        generate_narrative(_tampered_report())


def test_uses_mock_gateway_when_key_present(monkeypatch):
    called: dict[str, object] = {}

    def _fake_gateway(key, payload):
        called["key"] = key
        called["payload_findings"] = len(payload["findings"])
        return "Two independent signs of editing detected. Recommend reviewer inspection."

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    monkeypatch.setattr("tamper_detect.narrative._call_via_gateway", _fake_gateway)
    text = generate_narrative(_tampered_report())
    assert "editing" in text
    assert called["key"] == "test-key"
    assert called["payload_findings"] == 2


def test_uses_mock_anthropic_when_only_anthropic_key_present(monkeypatch):
    def _fake_anthropic(key, payload):
        return "Analyst summary."

    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anth-key")
    monkeypatch.setattr("tamper_detect.narrative._call_via_anthropic", _fake_anthropic)
    assert generate_narrative(_tampered_report()) == "Analyst summary."
