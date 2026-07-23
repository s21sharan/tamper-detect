"""Tier 3 — OCR-vs-render diff.

Re-render each page to a high-DPI raster and run tesseract on the pixels. If
the resulting OCR text disagrees materially with the reference text (either
the text layer extracted by pymupdf, or a caller-supplied OCR string), the
text layer is claiming something the pixels don't back up. That's the
signature of a text-under-image overlay OR an image splice on top of a
preserved text layer.

Signal name: text_layer_mismatches_render
"""

from __future__ import annotations

import re

import pytesseract

from tamper_detect.detectors.base import BaseDetector, DetectorContext, register
from tamper_detect.report import Finding


OCR_DPI = 300

# A token is any run of non-whitespace >= 2 chars. We normalize by lowercasing
# and stripping surrounding punctuation.
_TOKEN_RE = re.compile(r"\S{2,}")
_PUNCT_STRIP_RE = re.compile(r"^[^\w$]+|[^\w$]+$")


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(text or ""):
        norm = _PUNCT_STRIP_RE.sub("", raw).lower()
        if len(norm) < 2:
            continue
        out.add(norm)
    return out


# Tokens that look like they carry semantic weight for KYC (dollar amounts,
# EIN-like numbers, permit numbers, etc.).
_INTERESTING_RE = re.compile(
    r"""
    (?:^\$?\d[\d,]*(?:\.\d+)?$)      # currency or plain number
    | (?:^\d{2,}-\d{2,}(?:-\d+)?$)   # EIN / permit-style hyphenated ids
    | (?:^stp-)                      # sales tax permit prefix
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _significant(token: str) -> bool:
    return bool(_INTERESTING_RE.search(token))


class OcrRenderDiff(BaseDetector):
    name = "text_layer_mismatches_render"

    # If we find ≥ 1 "significant" token in the reference set that is missing
    # from the OCR (or vice versa), fire. Also fire for a large overall
    # divergence in noise-free token sets.
    _JACCARD_FIRE = 0.35

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        doc = ctx.doc

        # Reference text: caller-supplied if present, else the text layer.
        reference = ctx.supplied_ocr_text or doc.all_text()
        # If there's nothing to compare against (image-only PDF with no text
        # layer AND no supplied text), we can't fire on this signal.
        if not reference.strip():
            return []

        # OCR each page's render.
        ocr_parts: list[str] = []
        for page_num in range(doc.num_pages):
            img = doc.render_page(page_num, dpi=OCR_DPI)
            try:
                text = pytesseract.image_to_string(img)
            except Exception:
                # If tesseract can't be found we skip this detector cleanly.
                return []
            ocr_parts.append(text)
        ocr_text = "\n".join(ocr_parts).strip()
        if not ocr_text:
            # OCR failed / found nothing — no evidence either way.
            return []

        ref_tokens = _tokens(reference)
        ocr_tokens = _tokens(ocr_text)
        if not ref_tokens or not ocr_tokens:
            return []

        missing_from_ocr = ref_tokens - ocr_tokens
        extra_in_ocr = ocr_tokens - ref_tokens

        # Significant token diffs — the strongest signal.
        sig_missing = sorted(t for t in missing_from_ocr if _significant(t))
        sig_extra = sorted(t for t in extra_in_ocr if _significant(t))

        # Overall divergence — a broad noise floor.
        jaccard = len(ref_tokens & ocr_tokens) / max(1, len(ref_tokens | ocr_tokens))

        fires_sig = bool(sig_missing) or bool(sig_extra)
        fires_broad = jaccard < self._JACCARD_FIRE

        if not fires_sig and not fires_broad:
            return []

        # Score composition: significant differences dominate.
        score = 0.0
        if fires_sig:
            n = len(sig_missing) + len(sig_extra)
            score = min(1.0, 0.7 + 0.1 * n)
        if fires_broad:
            score = max(score, min(1.0, 1.0 - jaccard))

        return [
            Finding(
                signal=self.name,
                tier="ocr_diff",
                score=score,
                evidence={
                    "reference_source": "supplied" if ctx.supplied_ocr_text else "text_layer",
                    "jaccard_similarity": round(jaccard, 3),
                    "significant_missing_from_ocr": sig_missing[:20],
                    "significant_extra_in_ocr": sig_extra[:20],
                    "num_ref_tokens": len(ref_tokens),
                    "num_ocr_tokens": len(ocr_tokens),
                },
            )
        ]


register(OcrRenderDiff())
