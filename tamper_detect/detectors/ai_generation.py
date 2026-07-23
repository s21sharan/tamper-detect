"""Tier 4 — experimental AI-generation detectors.

These are intentionally lightweight heuristics. They exist so the pipeline has
*some* signal on fully-synthetic look-alike documents. Weight them low in
scoring — false-positive rates on real documents are unknown and likely high.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from tamper_detect.detectors.base import BaseDetector, DetectorContext, register
from tamper_detect.detectors.pixel_forensics import _page_scan_image, _rgb_to_np
from tamper_detect.report import Finding


AI_PRODUCER_TOKENS = (
    "diffusion",
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dalle",
    "generative",
    "ai-generated",
)


class AiGeneratedImage(BaseDetector):
    """Flag when an embedded raster page image has properties consistent with
    an image-generation model output:

      - Producer/Creator string names a known AI tool, OR
      - The image is the sole content of a page that "should" be a vector
        PDF, and its noise field is uniformly elevated (a common signature
        of low-QF re-encoded diffusion output).
    """

    name = "ai_generated_image"

    _UNIFORM_NOISE_STD = 0.9  # global std of the high-pass residual

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        info = ctx.doc.info_dict()
        producer = (info.producer or "").lower()
        creator = (info.creator or "").lower()

        matched_tokens: list[str] = []
        for tok in AI_PRODUCER_TOKENS:
            if tok in producer:
                matched_tokens.append(f"producer:{tok}")
            if tok in creator:
                matched_tokens.append(f"creator:{tok}")

        # Cheap producer/creator match — strongest evidence we have here.
        if matched_tokens:
            return [
                Finding(
                    signal=self.name,
                    tier="experimental",
                    score=0.85,
                    evidence={
                        "producer": info.producer,
                        "creator": info.creator,
                        "matched_tokens": matched_tokens,
                    },
                )
            ]

        # Fallback: image-only doc with uniformly elevated noise.
        text_layer = ctx.doc.all_text().strip()
        if text_layer:
            return []
        # Analyze page 1 only for POC.
        img, source = _page_scan_image(ctx.doc, 0)
        arr = _rgb_to_np(img).astype(np.float32).mean(axis=2)
        blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.5)
        residual = arr - blur
        global_std = float(np.std(residual))
        if global_std < self._UNIFORM_NOISE_STD:
            return []
        # Score capped at 0.6 — this heuristic is weak on its own.
        score = min(0.6, 0.4 + 0.2 * (global_std - self._UNIFORM_NOISE_STD))
        return [
            Finding(
                signal=self.name,
                tier="experimental",
                score=score,
                evidence={
                    "page": 0,
                    "source": source,
                    "global_residual_std": round(global_std, 3),
                    "reason": "image_only_with_elevated_noise",
                },
            )
        ]


class AiGeneratedText(BaseDetector):
    """Stub — very lightweight text-generation heuristic.

    For a real detector, you'd want a per-domain language model perplexity
    check. For the POC we just check for a couple of noticeably LLM-flavored
    boilerplate phrases; skip firing otherwise so we don't contribute noise.
    """

    name = "ai_generated_text"

    _SUSPICIOUS_PHRASES = (
        "as an ai language model",
        "i cannot generate",
        "let me know if you need",
    )

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        text = (ctx.supplied_ocr_text or ctx.doc.all_text()).lower()
        if not text:
            return []
        hits = [p for p in self._SUSPICIOUS_PHRASES if p in text]
        if not hits:
            return []
        return [
            Finding(
                signal=self.name,
                tier="experimental",
                score=0.7,
                evidence={"matched_phrases": hits},
            )
        ]


register(AiGeneratedImage())
register(AiGeneratedText())
