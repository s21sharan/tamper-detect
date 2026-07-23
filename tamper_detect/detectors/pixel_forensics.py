"""Tier 2 — pixel-domain forensics.

Signals produced:
  - ela_hotspots
  - jpeg_ghost
  - noise_residual_inconsistency
  - copy_move

Each detector works on a "scan image" per page. If the page has a large
embedded raster image (covers >~50% of the page area) we use that image
directly — preserving its original JPEG signature. Otherwise we render the
page at 200 DPI. That way vector PDFs get analyzed pixel-wise (mostly to find
copy-move on drawn elements) without inheriting fresh JPEG artifacts from the
rasterization step.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from tamper_detect.detectors.base import BaseDetector, DetectorContext, register
from tamper_detect.loader import LoadedDocument
from tamper_detect.report import Finding


PAGE_RENDER_DPI = 200
# Fraction of page area at which we prefer the embedded image over a fresh
# render.
_PAGE_IMAGE_COVERAGE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Scan-image selection
# ---------------------------------------------------------------------------


def _page_scan_image(doc: LoadedDocument, page_num: int) -> tuple[Image.Image, str]:
    """Return the "scan image" for the page and a source tag.

    Returns (image, source_tag), where source_tag is either "embedded" (we
    used the biggest embedded raster) or "rendered" (we rasterized).
    """
    page = doc._fitz_doc[page_num]  # noqa: SLF001
    page_area = page.rect.width * page.rect.height

    best: tuple[float, Image.Image] | None = None
    for emb in doc.embedded_images(page_num):
        if emb.bbox is None:
            continue
        x0, y0, x1, y1 = emb.bbox
        area = max(0.0, (x1 - x0) * (y1 - y0))
        coverage = area / page_area if page_area > 0 else 0.0
        if coverage < _PAGE_IMAGE_COVERAGE_THRESHOLD:
            continue
        if best is None or coverage > best[0]:
            best = (coverage, emb.image.convert("RGB"))
    if best is not None:
        return best[1], "embedded"

    return doc.render_page(page_num, dpi=PAGE_RENDER_DPI).convert("RGB"), "rendered"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgb_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)


def _resave_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blockwise_stat(
    arr: np.ndarray, block: int, fn
) -> np.ndarray:
    """Apply `fn` to each (block × block) tile of a 2-D array."""
    h, w = arr.shape[:2]
    nh = h // block
    nw = w // block
    out = np.zeros((nh, nw), dtype=np.float32)
    for i in range(nh):
        for j in range(nw):
            tile = arr[i * block : (i + 1) * block, j * block : (j + 1) * block]
            out[i, j] = fn(tile)
    return out


# ---------------------------------------------------------------------------
# 1. ELA — Error Level Analysis
# ---------------------------------------------------------------------------


class ElaHotspots(BaseDetector):
    """Compute an ELA residual against a fixed JPEG quality. Regions where the
    residual is anomalously high (relative to the median of the image) were
    likely saved at a lower quality than the rest of the image — a splice or
    edit signature.
    """

    name = "ela_hotspots"

    _ELA_QF = 75    # probe QF — resave at Q=75 to make JPEG artifacts visible
    _BLOCK = 32
    _CONTENT_ABS = 2.0      # blocks with residual above this are "content"
    _HOTSPOT_Z = 6.0        # only extreme outliers — text edges routinely
                            # produce z ≈ 3–4 on clean docs
    _MIN_CONTENT_BLOCKS = 20
    _MIN_HOT_BLOCKS = 8      # require a clustered region, not scattered edges

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        offenders: list[dict] = []
        for page_num in range(ctx.doc.num_pages):
            img, source = _page_scan_image(ctx.doc, page_num)
            if img.width < 100 or img.height < 100:
                continue
            resaved = _resave_jpeg(img, self._ELA_QF)
            a = _rgb_to_np(img).astype(np.int16)
            b = _rgb_to_np(resaved).astype(np.int16)
            residual = np.abs(a - b).mean(axis=2).astype(np.float32)

            # Blockwise mean residual, then filter to "content" blocks.
            block_mean = _blockwise_stat(residual, self._BLOCK, np.mean)
            content_mask = block_mean > self._CONTENT_ABS
            if content_mask.sum() < self._MIN_CONTENT_BLOCKS:
                continue
            content_vals = block_mean[content_mask]
            med = float(np.median(content_vals))
            mad = float(np.median(np.abs(content_vals - med))) or 1e-3
            zscores = (block_mean - med) / (1.4826 * mad)
            hot_mask = (zscores > self._HOTSPOT_Z) & content_mask
            num_hot = int(hot_mask.sum())
            frac_hot = num_hot / max(1, content_mask.sum())
            if num_hot < self._MIN_HOT_BLOCKS:
                continue
            offenders.append(
                {
                    "page": page_num,
                    "source": source,
                    "num_hot_blocks": num_hot,
                    "hot_fraction": round(frac_hot, 4),
                    "median_residual": round(med, 2),
                }
            )
        if not offenders:
            return []
        # Score grows with total hot-block area across pages.
        total_hot = sum(o["num_hot_blocks"] for o in offenders)
        score = min(1.0, 0.4 + 0.05 * total_hot)
        return [
            Finding(
                signal=self.name,
                tier="pixel",
                score=score,
                evidence={"pages": offenders},
            )
        ]


# ---------------------------------------------------------------------------
# 2. JPEG Ghost
# ---------------------------------------------------------------------------


class JpegGhost(BaseDetector):
    """Resave the image at several QFs; for each block, find which QF minimizes
    the residual. A block whose best-QF differs sharply from the global mode is
    a "ghost" — evidence it was originally compressed at a different QF than
    the rest of the image (a splice signature).
    """

    name = "jpeg_ghost"

    _QFS = (50, 60, 70, 80, 90)
    _BLOCK = 48
    _MIN_OUTLIER_BLOCKS = 3

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        offenders: list[dict] = []
        for page_num in range(ctx.doc.num_pages):
            img, source = _page_scan_image(ctx.doc, page_num)
            if source == "rendered":
                # No prior JPEG history to disagree with itself — every QF
                # gives the same residual profile. Skip.
                continue
            if img.width < 100 or img.height < 100:
                continue
            arr = _rgb_to_np(img).astype(np.float32).mean(axis=2)  # grayscale
            # residual[k] = |img - img_at_qf[k]| averaged over channels
            residuals = []
            for qf in self._QFS:
                resaved = _rgb_to_np(_resave_jpeg(img, qf)).astype(np.float32).mean(axis=2)
                residuals.append(np.abs(arr - resaved))
            # For each block, the QF index that minimizes mean residual
            h, w = arr.shape
            nh = h // self._BLOCK
            nw = w // self._BLOCK
            if nh == 0 or nw == 0:
                continue
            best_qf_idx = np.zeros((nh, nw), dtype=np.int32)
            for i in range(nh):
                for j in range(nw):
                    y0 = i * self._BLOCK
                    y1 = y0 + self._BLOCK
                    x0 = j * self._BLOCK
                    x1 = x0 + self._BLOCK
                    means = [r[y0:y1, x0:x1].mean() for r in residuals]
                    best_qf_idx[i, j] = int(np.argmin(means))

            # Mode across the whole image
            vals, counts = np.unique(best_qf_idx, return_counts=True)
            mode_idx = int(vals[int(np.argmax(counts))])
            # Blocks whose best QF is >= 2 steps away from the mode → outliers
            outliers = np.abs(best_qf_idx - mode_idx) >= 2
            num_out = int(outliers.sum())
            if num_out < self._MIN_OUTLIER_BLOCKS:
                continue
            offenders.append(
                {
                    "page": page_num,
                    "source": source,
                    "mode_qf": self._QFS[mode_idx],
                    "num_outlier_blocks": num_out,
                    "outlier_fraction": round(num_out / (nh * nw), 4),
                }
            )
        if not offenders:
            return []
        total = sum(o["num_outlier_blocks"] for o in offenders)
        score = min(1.0, 0.5 + 0.03 * total)
        return [
            Finding(
                signal=self.name,
                tier="pixel",
                score=score,
                evidence={"pages": offenders},
            )
        ]


# ---------------------------------------------------------------------------
# 3. Noise residual inconsistency
# ---------------------------------------------------------------------------


class NoiseResidualInconsistency(BaseDetector):
    """Split image into blocks; measure per-block sensor-noise proxy (std of
    high-pass residual). A splice breaks the noise field: a chunk of blocks
    with anomalous noise stats indicates content pasted from elsewhere.
    """

    name = "noise_residual_inconsistency"

    _BLOCK = 32
    _OUTLIER_Z = 3.5
    _CONTENT_STD = 0.3        # blocks with std above this are "content"
    _UNIFORM_NOISE_STD = 0.35  # if median-of-all-blocks std >= this, the image
                               # has noise applied uniformly — an AI-render tell

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        offenders: list[dict] = []
        for page_num in range(ctx.doc.num_pages):
            img, source = _page_scan_image(ctx.doc, page_num)
            if img.width < 100 or img.height < 100:
                continue
            arr = _rgb_to_np(img).astype(np.float32).mean(axis=2)
            blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.5)
            residual = arr - blur
            block_std = _blockwise_stat(residual, self._BLOCK, np.std)
            global_median = float(np.median(block_std))

            # Fire condition A: uniform elevated noise — the AI/diffusion tell.
            fires_uniform = global_median >= self._UNIFORM_NOISE_STD

            # Fire condition B: content blocks show outlier noise stats.
            content_mask = block_std > self._CONTENT_STD
            fires_outliers = False
            num_out = 0
            frac_out = 0.0
            if content_mask.sum() >= 20:
                content_vals = block_std[content_mask]
                med = float(np.median(content_vals))
                mad = float(np.median(np.abs(content_vals - med))) or 1e-3
                zscores = np.abs(block_std - med) / (1.4826 * mad)
                outliers = (zscores > self._OUTLIER_Z) & content_mask
                num_out = int(outliers.sum())
                frac_out = num_out / max(1, content_mask.sum())
                fires_outliers = num_out >= 8 and frac_out >= 0.05

            if not fires_uniform and not fires_outliers:
                continue
            offenders.append(
                {
                    "page": page_num,
                    "source": source,
                    "reason": "uniform_noise" if fires_uniform else "outlier_blocks",
                    "num_outlier_blocks": num_out,
                    "outlier_fraction": round(frac_out, 4),
                    "global_median_std": round(global_median, 3),
                }
            )
        if not offenders:
            return []
        # Uniform-noise firings get a moderate score; outlier-blocks scale with count.
        best = 0.4
        for o in offenders:
            if o["reason"] == "uniform_noise":
                best = max(best, min(1.0, 0.55 + 2.0 * (o["global_median_std"] - self._UNIFORM_NOISE_STD)))
            else:
                best = max(best, min(1.0, 0.4 + 0.02 * o["num_outlier_blocks"]))
        return [
            Finding(
                signal=self.name,
                tier="pixel",
                score=best,
                evidence={"pages": offenders},
            )
        ]


# ---------------------------------------------------------------------------
# 4. Copy-move
# ---------------------------------------------------------------------------


class CopyMove(BaseDetector):
    """Detect duplicated regions within the same page via ORB keypoints and
    self-matching. Clusters of matched keypoint pairs sharing the same
    displacement vector indicate copy-move.
    """

    name = "copy_move"

    _MIN_KEYPOINTS = 40
    _MATCH_RATIO = 0.75      # Lowe's ratio for filtering
    _MIN_DISPLACEMENT = 40   # pixels — filter self-matches
    _CLUSTER_TOL = 12        # pixels — displacement clustering tolerance
    _MIN_CLUSTER_SIZE = 8    # matches per cluster to count as copy-move

    def __call__(self, ctx: DetectorContext) -> list[Finding]:
        offenders: list[dict] = []
        for page_num in range(ctx.doc.num_pages):
            img, source = _page_scan_image(ctx.doc, page_num)
            if source == "rendered":
                # Vector text has too many repeated features for ORB
                # self-matching to be reliable — skip.
                continue
            if img.width < 200 or img.height < 200:
                continue
            gray = cv2.cvtColor(_rgb_to_np(img), cv2.COLOR_RGB2GRAY)
            orb = cv2.ORB_create(nfeatures=1500)
            kp, des = orb.detectAndCompute(gray, None)
            if des is None or len(kp) < self._MIN_KEYPOINTS:
                continue
            # k=2 self-matching
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des, des, k=3)
            # Skip the trivial self-match at k=0; take k=1 as best "other"
            candidate_pairs: list[tuple[int, int]] = []
            for m_group in matches:
                if len(m_group) < 2:
                    continue
                # m_group[0] is the query-to-self match; use m_group[1]
                best = m_group[1]
                # Lowe ratio against k=2 if we have it
                if len(m_group) >= 3 and best.distance / max(1.0, m_group[2].distance) > self._MATCH_RATIO:
                    continue
                q = kp[best.queryIdx].pt
                t = kp[best.trainIdx].pt
                dx = t[0] - q[0]
                dy = t[1] - q[1]
                if abs(dx) < self._MIN_DISPLACEMENT and abs(dy) < self._MIN_DISPLACEMENT:
                    continue
                candidate_pairs.append((int(dx), int(dy)))

            if len(candidate_pairs) < self._MIN_CLUSTER_SIZE:
                continue
            # Cluster displacements
            clusters: dict[tuple[int, int], int] = {}
            for dx, dy in candidate_pairs:
                key = (
                    int(round(dx / self._CLUSTER_TOL)) * self._CLUSTER_TOL,
                    int(round(dy / self._CLUSTER_TOL)) * self._CLUSTER_TOL,
                )
                clusters[key] = clusters.get(key, 0) + 1
            # Keep clusters with enough support
            good = [(k, v) for k, v in clusters.items() if v >= self._MIN_CLUSTER_SIZE]
            if not good:
                continue
            offenders.append(
                {
                    "page": page_num,
                    "source": source,
                    "clusters": [
                        {"displacement": [int(k[0]), int(k[1])], "matches": int(v)}
                        for k, v in sorted(good, key=lambda x: -x[1])[:5]
                    ],
                    "num_candidate_pairs": len(candidate_pairs),
                }
            )
        if not offenders:
            return []
        total = sum(c["matches"] for o in offenders for c in o["clusters"])
        score = min(1.0, 0.5 + 0.02 * total)
        return [
            Finding(
                signal=self.name,
                tier="pixel",
                score=score,
                evidence={"pages": offenders},
            )
        ]


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

DETECTORS = (
    ElaHotspots(),
    JpegGhost(),
    NoiseResidualInconsistency(),
    CopyMove(),
)

for _d in DETECTORS:
    register(_d)
