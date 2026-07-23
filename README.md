# tamper-detect

Forensics / tamper-detection layer for KYC PDF documents (utility invoices,
sales tax permits, EIN letters) uploaded during small-business account
onboarding.

Sits alongside an OCR pipeline and returns a per-document risk report with a
score, a decision (`pass` / `review` / `fail`), and an explainable list of
findings — each with the specific evidence that triggered it.

Design: [`docs/superpowers/specs/2026-07-23-tamper-detection-layer-design.md`](docs/superpowers/specs/2026-07-23-tamper-detection-layer-design.md).

---

## What it detects

| Tier | Signals | Catches |
|------|---------|---------|
| **1 — PDF structural** | `producer_creator_mismatch`, `mod_date_after_create_date`, `incremental_update_present`, `xref_anomaly`, `font_not_embedded`, `font_family_mix_in_line`, `text_over_image_overlay` | Documents saved through consumer PDF editors, edits made via incremental saves, mixed-font text overlays, "white-rect + new text" replacements. |
| **2 — Pixel forensics** | `ela_hotspots`, `jpeg_ghost`, `noise_residual_inconsistency`, `copy_move` | Doctored scans, spliced regions at different JPEG qualities, uniform AI-render noise, duplicated regions covering underlying content. |
| **3 — OCR vs render diff** | `text_layer_mismatches_render` | The text layer claims one thing, the pixels show another (overlays, image splices over preserved text). |
| **4 — Experimental** | `ai_generated_image`, `ai_generated_text`, `template_mismatch` | AI-flagged producer strings, image-only documents for issuers that normally send vector PDFs. Weighted low. |

Every finding contributes `tier_weight × signal_weight × normalized_score` to
the overall score. Thresholds live in [`tamper_detect/weights.yaml`](tamper_detect/weights.yaml)
so tuning doesn't require code changes.

## Install

Requires Python ≥ 3.11 and system `tesseract` (`brew install tesseract` on
macOS).

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Generate the test corpus

Produces 10 labelled PDFs (3 clean + 7 tampered) under `testdata/` along with
`labels.json` describing each document's ground truth.

```bash
python -m testdata.generate
```

## CLI

```bash
tamper-detect testdata/tampered/02_utility_invoice_amount_incremental.pdf
# → summary table, then decision (pass / review / fail).

tamper-detect testdata/tampered/09_ein_letter_ai_lookalike.pdf --json
# → full report as JSON.

tamper-detect path/to/bill.pdf --no-narrative
# → skip the Claude analyst summary.
```

## HTTP service

```bash
uvicorn tamper_detect.api:app --port 8000
```

Endpoints:

- `POST /analyze` — multipart PDF upload (+ optional `ocr_text` form field). Returns the report JSON.
- `POST /analyze-batch` — array of PDFs, array of reports back.
- `GET  /healthz` — trivial liveness.

```bash
curl -X POST http://localhost:8000/analyze \
  -F pdf=@testdata/tampered/02_utility_invoice_amount_incremental.pdf | jq
```

## Report shape

```json
{
  "document_id": "sha256:…",
  "doc_type_hint": "utility_invoice",
  "overall_score": 0.94,
  "decision": "fail",
  "findings": [
    {
      "signal": "incremental_update_present",
      "tier": "structural",
      "score": 0.85,
      "weight_applied": 0.8,
      "evidence": { "num_eof_markers": 2, "num_incremental_updates": 1 }
    }
  ],
  "narrative": null,
  "meta": {"detectors_run": ["producer_creator_mismatch", …], "runtime_ms": 1837, "version": "0.1.0"}
}
```

## Narrative (optional)

Set `ENABLE_NARRATIVE=1` and one of:

- `AI_GATEWAY_API_KEY` — routes through the Vercel AI Gateway (model via `NARRATIVE_MODEL`, default `anthropic/claude-sonnet-4-6`).
- `ANTHROPIC_API_KEY` — direct Anthropic call.

The narrative is reporting-only — never affects `overall_score` or `decision`.

## Tests

```bash
pytest                                       # everything (115 tests)
pytest tests/unit                            # per-detector + models
pytest tests/integration/test_corpus_eval.py -s   # evaluation harness with per-doc summary
```

## Evaluation on the 10-doc corpus

Current results — decision correctness: **10/10**.

| # | Doc | Kind | Expected signals | Fires |
|---|---|---|---|---|
| 1 | Utility invoice | clean | – | pass |
| 2 | Utility invoice | amount edit (incremental) | `incremental_update_present`, `text_layer_mismatches_render` | fail |
| 3 | Utility invoice | scanned + spliced amount | `jpeg_ghost`, `text_layer_mismatches_render` | fail |
| 4 | Sales tax permit | clean | – | pass |
| 5 | Sales tax permit | business-name overlay | `text_over_image_overlay` | fail |
| 6 | Sales tax permit | resave as Word | `producer_creator_mismatch` | review |
| 7 | EIN letter | clean | – | pass |
| 8 | EIN letter | EIN swap w/ mixed font (incremental) | `font_family_mix_in_line`, `incremental_update_present`, `text_layer_mismatches_render` | fail |
| 9 | EIN letter | AI-generated look-alike | `ai_generated_image`, `noise_residual_inconsistency`, `template_mismatch` | fail |
| 10 | Utility invoice | copy-move over watermark | `copy_move` | fail |

Rerun anytime with `pytest tests/integration/test_corpus_eval.py -s`.

## Layout

```
tamper_detect/
  analyze.py            # top-level entry point
  loader.py             # PDF parse layer
  detectors/            # one module per tier
    base.py             # protocol + registry + runner
    pdf_structural.py   # Tier 1
    pixel_forensics.py  # Tier 2
    ocr_render_diff.py  # Tier 3
    ai_generation.py    # Tier 4
    template_match.py   # Tier 4
  scorer.py             # weighted combine + decision
  narrative.py          # Claude via AI Gateway
  report.py             # pydantic Finding / Report
  weights.yaml
  cli.py
  api.py
testdata/
  pdf_builders.py       # clean-doc templates
  tamper_ops.py         # tampering operations
  generate.py           # build corpus + labels.json
tests/
  unit/                 # per-module tests
  integration/          # full-corpus evaluation harness
```

## Known limitations (POC)

- Synthetic test corpus only — real-world fraud will look different. Tune weights against a labelled real corpus before production.
- `ela_hotspots` is intentionally conservative; JPEG Ghost is the primary splice detector.
- `text_layer_mismatches_render` can be defeated when the swap preserves shared tokens (e.g., business name that appears in multiple places). Covered by other detectors.
- No auth on the FastAPI service — prototype only.
- In-memory analysis; no PII / retention story yet.
