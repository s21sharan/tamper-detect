"""Integration harness — run analyze() against every test-corpus document and
assert the labels are respected. Prints a confusion-matrix-style summary at
the end so weights can be tuned by inspection.

The pass criteria are intentionally lenient:
- Every clean document should get decision == "pass"
- Every tampered document should get decision != "pass"
- Every "expected_signal" listed in labels.json should fire with score > 0
  on the tampered document it belongs to (soft — printed but only asserted
  for a subset of the strongest signals)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tamper_detect.analyze import analyze


REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO_ROOT / "testdata" / "labels.json"


# Signals we insist must fire on their labelled documents (the deterministic,
# high-confidence ones). Weaker/experimental signals — and content-diff
# signals that can be fooled by shared tokens between old and new text — are
# checked in the printed report but not asserted.
STRONG_SIGNALS = {
    "incremental_update_present",
    "producer_creator_mismatch",
    "text_over_image_overlay",
    "font_family_mix_in_line",
    "copy_move",
    "template_mismatch",
}


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    assert LABELS_PATH.exists(), "run: python -m testdata.generate"
    return json.loads(LABELS_PATH.read_text())["docs"]


@pytest.fixture(scope="module")
def results(corpus: list[dict]) -> list[tuple[dict, object]]:
    out = []
    for entry in corpus:
        pdf = (REPO_ROOT / entry["file"]).read_bytes()
        report = analyze(pdf, enable_narrative=False)
        out.append((entry, report))
    return out


def test_clean_documents_pass(results):
    failing_cleans = [
        (e["file"], r.decision, r.overall_score)
        for (e, r) in results
        if e["authentic"] and r.decision != "pass"
    ]
    assert not failing_cleans, f"clean docs should pass: {failing_cleans}"


def test_tampered_documents_are_flagged(results):
    passing_tampered = [
        (e["file"], r.decision, r.overall_score)
        for (e, r) in results
        if not e["authentic"] and r.decision == "pass"
    ]
    assert not passing_tampered, f"tampered docs should not pass: {passing_tampered}"


def test_strong_signals_fire_on_labelled_docs(results):
    misses: list[tuple[str, str]] = []
    for entry, report in results:
        if entry["authentic"]:
            continue
        fired = {f.signal for f in report.findings}
        for sig in entry["expected_signals"]:
            if sig in STRONG_SIGNALS and sig not in fired:
                misses.append((entry["file"], sig))
    assert not misses, f"expected strong signals missed: {misses}"


def test_print_evaluation_summary(results, capsys):
    """Prints a per-doc summary for eyeball evaluation. Never fails."""
    with capsys.disabled():
        print()
        print("=" * 88)
        print("Tamper detection — evaluation on 10-document corpus")
        print("=" * 88)
        n_docs = len(results)
        n_correct = 0
        for entry, report in results:
            expected_pass = entry["authentic"]
            actual_pass = report.decision == "pass"
            correct = expected_pass == actual_pass
            n_correct += int(correct)
            mark = "OK" if correct else "MISS"
            print(
                f"[{mark:>4}] {Path(entry['file']).name:52} "
                f"score={report.overall_score:.2f} "
                f"decision={report.decision:6} "
                f"({'clean' if expected_pass else 'tampered'})"
            )
            fired = {f.signal for f in report.findings}
            expected = set(entry["expected_signals"])
            if expected:
                print(f"       expected: {sorted(expected)}")
                print(f"       fired:    {sorted(fired)}")
                missing = expected - fired
                extra = fired - expected
                if missing:
                    print(f"       missing:  {sorted(missing)}")
                if extra:
                    print(f"       extra:    {sorted(extra)}")
            print()
        print("=" * 88)
        print(f"Overall decision correctness: {n_correct}/{n_docs}")
        print("=" * 88)
