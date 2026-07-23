"""Tier 1 — PDF structural forensics.

These detectors read /Info, xref, fonts, and text span metadata. They're
deterministic, cheap, and high-precision — the first line of defense.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Iterable

from tamper_detect.detectors.base import BaseDetector, DetectorContext, register
from tamper_detect.loader import LoadedDocument, TextSpan
from tamper_detect.report import Finding


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The Standard 14 PDF fonts do not need to be embedded per PDF spec.
STANDARD_14_FONTS = {
    "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
    "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
    "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique",
    "Symbol", "ZapfDingbats",
}

# Tokens that appear in the Producer/Creator strings of common consumer
# PDF-editing tools. Their presence in a KYC document is a mild-to-strong
# signal it was edited outside the issuing system.
CONSUMER_EDITOR_TOKENS = (
    "ilovepdf",
    "smallpdf",
    "sejda",
    "pdfescape",
    "pdf-xchange",
    "pdfelement",
    "nitro pdf",
    "nitro pro",
    "foxit phantompdf",
    "soda pdf",
    "sodapdf",
    "adobe acrobat pro",  # generic "Adobe Acrobat" is fine, "Pro" often means editing
    "microsoft word",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PDF_DATE_RE = re.compile(r"^D?:?(\d{14})")


def _parse_pdf_date(raw: str | None) -> datetime | None:
    """Parse a PDF date string like `D:20260614120000+00'00'` → datetime.

    Returns None on any failure. Timezone is ignored — comparison is purely on
    the naive datetime, which is fine for our "did mod_date follow create_date"
    question since both come from the same file.
    """
    if not raw:
        return None
    m = _PDF_DATE_RE.match(raw)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _font_family_base(font_name: str) -> str:
    """Reduce a PDF font name to its family base.

    Strips a subset prefix ('ABCDEF+') and everything after the first '-' or ','.
    Example: 'ABCDEF+Helvetica-Bold' → 'Helvetica'.
    """
    name = font_name
    if "+" in name:
        prefix, rest = name.split("+", 1)
        # Real subset prefixes are exactly 6 uppercase letters.
        if len(prefix) == 6 and prefix.isupper() and prefix.isalpha():
            name = rest
    for sep in ("-", ","):
        if sep in name:
            name = name.split(sep, 1)[0]
    return name.strip()


def _bbox_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _bbox_area(b: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = b
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _is_near_white(color: tuple[float, ...] | None) -> bool:
    if not color:
        return False
    # fitz colors are in [0, 1] as tuples (r, g, b) or (gray,).
    if len(color) == 1:
        return color[0] > 0.95
    if len(color) >= 3:
        return all(c > 0.95 for c in color[:3])
    return False


# ---------------------------------------------------------------------------
# Detector 1: producer / creator mismatch
# ---------------------------------------------------------------------------


class ProducerCreatorMismatch(BaseDetector):
    name = "producer_creator_mismatch"

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        info = ctx.doc.info_dict()
        producer = (info.producer or "").strip()
        creator = (info.creator or "").strip()
        matched: list[str] = []
        for token in CONSUMER_EDITOR_TOKENS:
            if token in producer.lower():
                matched.append(f"producer contains {token!r}")
            if token in creator.lower():
                matched.append(f"creator contains {token!r}")
        # A distinct Producer and Creator on a KYC document is suspicious when
        # they name different products.
        producer_lc = producer.lower()
        creator_lc = creator.lower()
        distinct_pair = (
            producer and creator
            and producer_lc != creator_lc
            and producer_lc.split()[0] != creator_lc.split()[0]
        )
        if not matched and not distinct_pair:
            return []
        # Score: high if a consumer-editor token matched; moderate for a mere
        # distinct-product pair.
        score = 0.9 if matched else 0.5
        return [
            Finding(
                signal=self.name,
                tier="structural",
                score=score,
                evidence={
                    "producer": producer or None,
                    "creator": creator or None,
                    "matched_tokens": matched,
                    "distinct_product_pair": distinct_pair,
                },
            )
        ]


# ---------------------------------------------------------------------------
# Detector 2: mod_date after create_date
# ---------------------------------------------------------------------------


class ModDateAfterCreateDate(BaseDetector):
    name = "mod_date_after_create_date"

    # A save typically writes both dates within the same second — treat sub-
    # minute deltas as noise, not a real modification.
    _MIN_DELTA = timedelta(minutes=1)

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        info = ctx.doc.info_dict()
        created = _parse_pdf_date(info.creation_date)
        modified = _parse_pdf_date(info.mod_date)
        if created is None or modified is None:
            return []
        delta = modified - created
        if delta < self._MIN_DELTA:
            return []
        # Map delta to a score: 1 min → 0.3, 1 hour → 0.6, 1 day → 0.8, 30 day+ → 0.95
        hours = delta.total_seconds() / 3600.0
        if hours < 1:
            score = 0.3 + 0.3 * (hours / 1.0)
        elif hours < 24:
            score = 0.6 + 0.2 * ((hours - 1) / 23.0)
        elif hours < 24 * 30:
            score = 0.8 + 0.15 * ((hours - 24) / (24 * 29))
        else:
            score = 0.95
        return [
            Finding(
                signal=self.name,
                tier="structural",
                score=min(1.0, score),
                evidence={
                    "creation_date": info.creation_date,
                    "mod_date": info.mod_date,
                    "delta_seconds": int(delta.total_seconds()),
                },
            )
        ]


# ---------------------------------------------------------------------------
# Detector 3: incremental updates present
# ---------------------------------------------------------------------------


class IncrementalUpdatePresent(BaseDetector):
    name = "incremental_update_present"

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        summary = ctx.doc.xref_summary()
        if not summary.has_incremental_updates:
            return []
        # More %%EOF markers → more updates → stronger signal
        extras = max(0, summary.num_eof_markers - 1)
        score = min(1.0, 0.7 + 0.15 * extras)
        return [
            Finding(
                signal=self.name,
                tier="structural",
                score=score,
                evidence={
                    "num_eof_markers": summary.num_eof_markers,
                    "num_incremental_updates": extras,
                },
            )
        ]


# ---------------------------------------------------------------------------
# Detector 4: xref anomaly (very conservative stub)
# ---------------------------------------------------------------------------


class XrefAnomaly(BaseDetector):
    """Fires only if the PDF fails a basic sanity check we already learn from
    pymupdf/pikepdf. Kept lean — the file wouldn't have loaded at all if the
    xref were badly broken."""

    name = "xref_anomaly"

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        summary = ctx.doc.xref_summary()
        # Heuristic: extremely few objects on a multi-page doc is odd.
        if ctx.doc.num_pages > 1 and summary.num_objects < 3:
            return [
                Finding(
                    signal=self.name,
                    tier="structural",
                    score=0.6,
                    evidence={"num_objects": summary.num_objects, "num_pages": ctx.doc.num_pages},
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Detector 5: font not embedded (excludes Standard 14)
# ---------------------------------------------------------------------------


class FontNotEmbedded(BaseDetector):
    name = "font_not_embedded"

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        non_std_missing: list[str] = []
        for f in ctx.doc.fonts():
            # Use basefont (real font name) for standard-14 comparison, not the
            # local resource alias.
            candidates = {f.basefont, _font_family_base(f.basefont), f.name}
            if any(c in STANDARD_14_FONTS for c in candidates if c):
                continue
            if not f.embedded:
                non_std_missing.append(f.basefont or f.name)
        if not non_std_missing:
            return []
        score = min(1.0, 0.5 + 0.2 * len(non_std_missing))
        return [
            Finding(
                signal=self.name,
                tier="structural",
                score=score,
                evidence={"fonts_not_embedded": non_std_missing},
            )
        ]


# ---------------------------------------------------------------------------
# Detector 6: font family mix in a single line
# ---------------------------------------------------------------------------


class FontFamilyMixInLine(BaseDetector):
    """Fires when a line contains multiple font families, OR when spans on
    different fitz-detected lines have spatially overlapping bboxes but
    different families (the canonical overlay pattern: a new span drawn on
    top of an old span at a slightly different y-baseline).
    """

    name = "font_family_mix_in_line"

    # Ignore very short lines — punctuation-only fragments etc.
    _MIN_LINE_CHARS = 3
    # Fraction of the smaller bbox that must overlap to count as "same line."
    _OVERLAP_FRAC = 0.3

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        offenders: list[dict] = []

        # Pass 1: within-line font-family mixing.
        for page_num in range(ctx.doc.num_pages):
            for line in ctx.doc.text_lines(page_num):
                total_text = "".join(s.text for s in line).strip()
                if len(total_text) < self._MIN_LINE_CHARS:
                    continue
                families = {_font_family_base(s.font) for s in line if s.font}
                families.discard("")
                if len(families) >= 2:
                    offenders.append(
                        {
                            "page": page_num,
                            "kind": "within_line",
                            "text": total_text[:80],
                            "families": sorted(families),
                        }
                    )

        # Pass 2: spatial overlap between spans on different lines with
        # different font families — the "text over text" overlay signature.
        for page_num in range(ctx.doc.num_pages):
            spans = [
                s for s in ctx.doc.text_spans(page_num) if s.text.strip() and s.font
            ]
            for i, a in enumerate(spans):
                for b in spans[i + 1 :]:
                    if _font_family_base(a.font) == _font_family_base(b.font):
                        continue
                    if len(a.text.strip()) < self._MIN_LINE_CHARS:
                        continue
                    if len(b.text.strip()) < self._MIN_LINE_CHARS:
                        continue
                    overlap = _bbox_overlap_area(a.bbox, b.bbox)
                    if overlap <= 0:
                        continue
                    smaller = min(_bbox_area(a.bbox), _bbox_area(b.bbox))
                    if smaller <= 0:
                        continue
                    if overlap / smaller >= self._OVERLAP_FRAC:
                        offenders.append(
                            {
                                "page": page_num,
                                "kind": "spatial_overlap",
                                "text_a": a.text[:60],
                                "text_b": b.text[:60],
                                "families": sorted(
                                    [_font_family_base(a.font), _font_family_base(b.font)]
                                ),
                            }
                        )
        if not offenders:
            return []
        score = min(1.0, 0.75 + 0.05 * len(offenders))
        return [
            Finding(
                signal=self.name,
                tier="structural",
                score=score,
                evidence={"offending_lines": offenders[:20], "count": len(offenders)},
            )
        ]


# ---------------------------------------------------------------------------
# Detector 7: text over image / white-fill overlay
# ---------------------------------------------------------------------------


class TextOverImageOverlay(BaseDetector):
    """Fires when a text span sits on top of a near-white filled shape.

    This is the canonical signature of "white out the old text, write the new
    text on top." The old text stream is usually left behind — combined with
    ocr_render_diff, this is a very strong tamper signal.
    """

    name = "text_over_image_overlay"

    # Fraction of the text bbox that must be inside a fill to count.
    _MIN_OVERLAP_FRAC = 0.5

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        import fitz  # local import so the module has no hard fitz dep at load

        doc = ctx.doc
        offenders: list[dict] = []
        for page_num in range(doc.num_pages):
            page = doc._fitz_doc[page_num]  # noqa: SLF001
            fills: list[tuple[float, float, float, float]] = []
            try:
                drawings = page.get_drawings()
            except Exception:
                drawings = []
            for d in drawings:
                if d.get("type") not in ("f", "fs"):
                    # Only interested in filled paths.
                    continue
                if not _is_near_white(d.get("fill")):
                    continue
                rect = d.get("rect")
                if rect is None:
                    continue
                fills.append(tuple(rect))  # type: ignore[arg-type]

            if not fills:
                continue

            for span in doc.text_spans(page_num):
                area = _bbox_area(span.bbox)
                if area <= 0.0 or not span.text.strip():
                    continue
                for f_rect in fills:
                    overlap = _bbox_overlap_area(span.bbox, f_rect)
                    if overlap / area >= self._MIN_OVERLAP_FRAC:
                        offenders.append(
                            {
                                "page": page_num,
                                "text": span.text[:80],
                                "bbox": list(span.bbox),
                                "overlap_frac": overlap / area,
                            }
                        )
                        break

        if not offenders:
            return []
        score = min(1.0, 0.75 + 0.05 * len(offenders))
        return [
            Finding(
                signal=self.name,
                tier="structural",
                score=score,
                evidence={"offenders": offenders[:20], "count": len(offenders)},
            )
        ]


# ---------------------------------------------------------------------------
# Register all Tier 1 detectors
# ---------------------------------------------------------------------------


DETECTORS = (
    ProducerCreatorMismatch(),
    ModDateAfterCreateDate(),
    IncrementalUpdatePresent(),
    XrefAnomaly(),
    FontNotEmbedded(),
    FontFamilyMixInLine(),
    TextOverImageOverlay(),
)

for _d in DETECTORS:
    register(_d)


def all_detectors() -> Iterable[BaseDetector]:
    return DETECTORS
