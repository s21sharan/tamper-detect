"""Narrative generation — hand the deterministic report to Claude and get back
a 3–5 sentence analyst summary. Reporting-only: never affects overall_score
or decision.

Uses the Vercel AI Gateway when AI_GATEWAY_API_KEY is set; falls back to
direct Anthropic API when ANTHROPIC_API_KEY is set instead.
"""

from __future__ import annotations

import json
import os

from tamper_detect.report import Report

DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


_SYSTEM = (
    "You are a document forensics analyst. Given a deterministic report of "
    "tamper-detection findings on a KYC document, produce a 3-5 sentence "
    "plain-English summary suitable for another analyst. Do not invent "
    "evidence. Do not restate the score verbatim. Explain what each finding "
    "means in practical terms and what a reviewer should look for."
)


def generate_narrative(report: Report) -> str:
    """Return a short analyst-style summary of the report.

    Raises RuntimeError if no credentials are available.
    """
    payload = {
        "overall_score": report.overall_score,
        "decision": report.decision,
        "doc_type_hint": report.doc_type_hint,
        "findings": [
            {
                "signal": f.signal,
                "tier": f.tier,
                "score": f.score,
                "weight_applied": f.weight_applied,
                "evidence": f.evidence,
            }
            for f in report.findings
        ],
    }

    gateway_key = os.getenv("AI_GATEWAY_API_KEY")
    if gateway_key:
        return _call_via_gateway(gateway_key, payload)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        return _call_via_anthropic(anthropic_key, payload)

    raise RuntimeError(
        "no AI credentials — set AI_GATEWAY_API_KEY or ANTHROPIC_API_KEY"
    )


def _call_via_gateway(api_key: str, payload: dict) -> str:  # pragma: no cover
    import httpx

    model = os.getenv("NARRATIVE_MODEL", DEFAULT_MODEL)
    url = "https://ai-gateway.vercel.sh/v1/chat/completions"
    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_via_anthropic(api_key: str, payload: dict) -> str:  # pragma: no cover
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    model = os.getenv("NARRATIVE_MODEL", "claude-sonnet-4-6")
    msg = client.messages.create(
        model=model,
        max_tokens=300,
        system=_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    parts: list[str] = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()
