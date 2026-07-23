"""Tampering operations used to produce the tampered half of the test corpus.

Each op takes clean PDF bytes and returns tampered PDF bytes, plus a short
`tamper_kind` label. The operations are intentionally chosen to trigger
specific detector signals — see docstrings.
"""

from __future__ import annotations

import io
import os
import random
import tempfile

import fitz  # pymupdf
import numpy as np
import pikepdf
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _save_incremental_via_fitz(doc: fitz.Document, path: str) -> None:
    """Save `doc` incrementally in-place — appends a second %%EOF marker."""
    doc.save(path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)


def _find_text_rect(page: fitz.Page, needle: str) -> fitz.Rect | None:
    """Return the first bbox of a search hit for `needle` on `page`."""
    for rect in page.search_for(needle):
        return rect
    return None


def _find_text_rects(page: fitz.Page, needle: str) -> list[fitz.Rect]:
    """Return every bbox of `needle` on `page`."""
    return list(page.search_for(needle))


def _white_out_and_write(
    page: fitz.Page,
    rect: fitz.Rect,
    new_text: str,
    font_size: float = 11.0,
    fontname: str = "helv",
    pad: float = 1.5,
) -> None:
    """Draw a white filled rect over `rect` and write `new_text` on top.

    Note: does NOT remove the old text stream operators — the text layer will
    still contain the old text, which is what makes the OCR-render diff fire.
    """
    padded = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
    page.draw_rect(padded, color=(1, 1, 1), fill=(1, 1, 1))
    y = rect.y1 - 2  # baseline near the bottom of the original rect
    page.insert_text((rect.x0, y), new_text, fontname=fontname, fontsize=font_size)


# ---------------------------------------------------------------------------
# 2. Amount edited via incremental update
# ---------------------------------------------------------------------------


def edit_amount_incremental(clean_pdf: bytes, old_amount: str, new_amount: str) -> bytes:
    """Overlay `new_amount` on `old_amount` in the first page, save incrementally.

    Triggers: incremental_update_present, mod_date_after_create_date,
    text_layer_mismatches_render (text layer still has old_amount), and often
    font_family_mix_in_line (Helvetica default vs Helvetica-Bold in the original).
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(clean_pdf)
        path = tf.name
    try:
        doc = fitz.open(path)
        page = doc[0]
        rect = _find_text_rect(page, old_amount)
        if rect is None:
            raise ValueError(f"could not find {old_amount!r} in the PDF")
        _white_out_and_write(page, rect, new_amount, font_size=13.0, fontname="helv")
        doc.set_metadata({"title": "edited"})
        _save_incremental_via_fitz(doc, path)
        doc.close()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 3. Scanned + spliced amount (image-only PDF with a splice at wrong JPEG QF)
# ---------------------------------------------------------------------------


def scan_and_splice_amount(
    clean_pdf: bytes,
    old_amount: str,
    new_amount: str,
    scan_qf: int = 88,
    splice_qf: int = 62,
) -> bytes:
    """Rasterize page 1, splice the amount region at a different JPEG quality,
    then wrap the image OVER the original page (keeping the text layer).

    Triggers: jpeg_ghost, ela_hotspots, noise_residual_inconsistency, and
    text_layer_mismatches_render (image says new, text layer says old).
    """
    src_doc = fitz.open(stream=clean_pdf, filetype="pdf")
    src_page = src_doc[0]

    # 1. Render page to raster at scan-like DPI.
    pix = src_page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
    page_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # 2. Save/reload as JPEG at scan_qf so the whole page has QF ~= scan_qf.
    buf = io.BytesIO()
    page_img.save(buf, format="JPEG", quality=scan_qf)
    buf.seek(0)
    page_img = Image.open(buf).convert("RGB")

    # 3. Draw the new amount over the old amount region, then re-save just that
    #    region at splice_qf to bake in a second quantization signature.
    rect = _find_text_rect(src_page, old_amount)
    if rect is None:
        raise ValueError(f"could not find {old_amount!r}")
    zoom = 200 / 72
    x0 = int(rect.x0 * zoom) - 4
    y0 = int(rect.y0 * zoom) - 4
    x1 = int(rect.x1 * zoom) + 40  # wider for the longer new amount
    y1 = int(rect.y1 * zoom) + 6

    # Splice region: white rect + rendered text
    splice = page_img.crop((x0, y0, x1, y1)).copy()
    draw = ImageDraw.Draw(splice)
    draw.rectangle((0, 0, splice.width, splice.height), fill=(255, 255, 255))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 2), new_amount, fill=(0, 0, 0), font=font)

    # Re-save splice at splice_qf, reload, paste back.
    sbuf = io.BytesIO()
    splice.save(sbuf, format="JPEG", quality=splice_qf)
    sbuf.seek(0)
    splice_reloaded = Image.open(sbuf).convert("RGB")
    page_img.paste(splice_reloaded, (x0, y0))

    # 4. Overlay the modified image onto the original page (keeps text layer).
    out_doc = fitz.open()
    new_page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
    # Bake the source page content (text layer) into out_doc.
    new_page.show_pdf_page(new_page.rect, src_doc, 0)
    # Now insert the tampered image over it.
    img_buf = io.BytesIO()
    page_img.save(img_buf, format="JPEG", quality=scan_qf)
    new_page.insert_image(new_page.rect, stream=img_buf.getvalue())

    out_bytes = out_doc.tobytes()
    out_doc.close()
    src_doc.close()
    return out_bytes


# ---------------------------------------------------------------------------
# 5. Name/business swap via text-layer overlay (not removing old text)
# ---------------------------------------------------------------------------


def overlay_swap_text(clean_pdf: bytes, old_text: str, new_text: str) -> bytes:
    """Draw a white rect over EVERY occurrence of `old_text` and stamp
    `new_text` on top of each.

    The original text stream is untouched, so the text layer still contains
    `old_text`. Triggers: text_over_image_overlay (fill on top of a text region),
    text_layer_mismatches_render (rendered/OCR text != text-layer text).
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(clean_pdf)
        path = tf.name
    try:
        doc = fitz.open(path)
        page = doc[0]
        rects = _find_text_rects(page, old_text)
        if not rects:
            raise ValueError(f"could not find {old_text!r}")
        for r in rects:
            font_size = max(8.0, min(14.0, r.height * 0.9))
            _white_out_and_write(page, r, new_text, font_size=font_size)
        out = doc.tobytes()  # fresh save, no incremental
        doc.close()
        return out
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 6. Re-save through a Word-like path (Producer/Creator mismatch)
# ---------------------------------------------------------------------------


def resave_as_word(clean_pdf: bytes) -> bytes:
    """Rewrite the /Info dict so Producer and Creator disagree suspiciously."""
    with pikepdf.open(io.BytesIO(clean_pdf)) as p:
        with p.open_metadata() as m:
            m["dc:title"] = "Sales Tax Permit"
        # Overwrite low-level docinfo
        p.docinfo["/Creator"] = pikepdf.String("Microsoft Word for Microsoft 365")
        p.docinfo["/Producer"] = pikepdf.String("iLovePDF (Free)")
        buf = io.BytesIO()
        p.save(buf, static_id=True)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# 8. EIN digits replaced via incremental save with a mixed font
# ---------------------------------------------------------------------------


def swap_ein_incremental(clean_pdf: bytes, old_ein: str, new_ein: str) -> bytes:
    """Overlay new EIN digits in Times (mixed with the original Courier) at
    every occurrence, and save incrementally.

    Triggers: font_family_mix_in_line, incremental_update_present,
    text_layer_mismatches_render.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(clean_pdf)
        path = tf.name
    try:
        doc = fitz.open(path)
        page = doc[0]
        rects = _find_text_rects(page, old_ein)
        if not rects:
            raise ValueError(f"could not find {old_ein!r}")
        for r in rects:
            # Match the visual size of the surrounding text — the header EIN
            # is larger than body-text EINs. Use the rect height as a proxy.
            font_size = max(9.0, min(16.0, r.height * 0.9))
            _white_out_and_write(page, r, new_ein, font_size=font_size, fontname="tiro")
        doc.set_metadata({"title": "modified EIN"})
        _save_incremental_via_fitz(doc, path)
        doc.close()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 9. AI-generated look-alike EIN (render + noise + JPEG artifact)
# ---------------------------------------------------------------------------


def ai_lookalike(clean_pdf: bytes, noise_sigma: float = 4.0, jpeg_qf: int = 55) -> bytes:
    """Render clean PDF, add Gaussian noise + low-QF JPEG artifacts, then wrap
    as an image-only PDF (no text layer). Simulates a fully synthetic look-alike.

    Triggers: noise_residual_inconsistency (whole page has aggressive noise
    signature), template_mismatch (layout is close but not identical), and
    ai_generated_image (heuristic scan of raster page).
    """
    src_doc = fitz.open(stream=clean_pdf, filetype="pdf")
    src_page = src_doc[0]
    pix = src_page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
    page_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    arr = np.array(page_img).astype(np.float32)
    rng = np.random.default_rng(seed=0xA1)
    noise = rng.normal(0, noise_sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr)

    buf = io.BytesIO()
    noisy.save(buf, format="JPEG", quality=jpeg_qf)
    buf.seek(0)
    lossy = Image.open(buf).convert("RGB")

    out_doc = fitz.open()
    new_page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
    img_buf = io.BytesIO()
    lossy.save(img_buf, format="JPEG", quality=jpeg_qf)
    new_page.insert_image(new_page.rect, stream=img_buf.getvalue())
    out_doc.set_metadata(
        {"creator": "diffusion-doc-gen v0.4", "producer": "diffusion-doc-gen v0.4"}
    )
    out_bytes = out_doc.tobytes()
    out_doc.close()
    src_doc.close()
    return out_bytes


# ---------------------------------------------------------------------------
# 10. Copy-moved logo covering an underlying watermark
# ---------------------------------------------------------------------------


def copy_move_logo(clean_pdf: bytes) -> bytes:
    """Render page, first stamp a 'PAID' watermark, then duplicate a small
    corner region and paste it over the watermark. Wrap as image-only PDF.

    Triggers: copy_move, ela_hotspots.
    """
    src_doc = fitz.open(stream=clean_pdf, filetype="pdf")
    src_page = src_doc[0]
    pix = src_page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False)
    page_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGB")

    # Stamp a big grey "PAID" across the middle.
    draw = ImageDraw.Draw(page_img)
    try:
        wm_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
    except Exception:
        wm_font = ImageFont.load_default()
    draw.text((page_img.width // 3, page_img.height // 2 - 40), "PAID", fill=(200, 200, 200), font=wm_font)

    # Grab a chunk of clean background from the bottom margin.
    strip_h = 120
    src_box = (60, page_img.height - strip_h - 20, page_img.width - 60, page_img.height - 20)
    strip = page_img.crop(src_box)

    # Paste it OVER the middle band to cover PAID.
    dst_top = page_img.height // 2 - strip_h // 2
    page_img.paste(strip, (60, dst_top))

    # Duplicate a smaller distinctive tile from another region.
    tile_src = (72, page_img.height - 180, 200, page_img.height - 100)
    tile = page_img.crop(tile_src)
    page_img.paste(tile, (400, page_img.height - 300))

    out_doc = fitz.open()
    new_page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
    buf = io.BytesIO()
    page_img.save(buf, format="JPEG", quality=90)
    new_page.insert_image(new_page.rect, stream=buf.getvalue())
    out_bytes = out_doc.tobytes()
    out_doc.close()
    src_doc.close()
    return out_bytes


# Deterministic seeding for anything that uses global random.
random.seed(0xC0FFEE)
