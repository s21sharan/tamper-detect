"""Tests for the DocumentLoader.

Builds tiny test PDFs in-memory with reportlab so we don't depend on fixture
files. The test-corpus module later reuses these building blocks.
"""

from __future__ import annotations

import io
import os
import tempfile

import fitz
import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from tamper_detect.loader import LoadedDocument, load


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_simple_pdf(text: str = "hello world", pages: int = 1) -> bytes:
    """Build a minimal reportlab PDF with one text line per page."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    for i in range(pages):
        c.setFont("Helvetica", 12)
        c.drawString(72, 720, f"{text} (page {i + 1})")
        c.showPage()
    c.save()
    return buf.getvalue()


def _apply_incremental_update(pdf_bytes: bytes) -> bytes:
    """Do a real PDF incremental save via pymupdf — appends a second %%EOF."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(pdf_bytes)
        path = tf.name
    try:
        doc = fitz.open(path)
        doc.set_metadata({"title": "edited"})
        doc.save(path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        doc.close()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestLoaderBasics:
    def test_load_reportlab_pdf(self):
        b = _make_simple_pdf("test")
        with load(b) as doc:
            assert doc.num_pages == 1
            assert doc.document_id.startswith("sha256:")
            assert len(doc.document_id) == len("sha256:") + 64

    def test_multipage(self):
        b = _make_simple_pdf(pages=3)
        with load(b) as doc:
            assert doc.num_pages == 3

    def test_document_id_is_content_hash(self):
        b1 = _make_simple_pdf("alpha")
        b2 = _make_simple_pdf("beta")
        with load(b1) as d1, load(b2) as d2:
            assert d1.document_id != d2.document_id

    def test_empty_bytes_rejected(self):
        with pytest.raises(ValueError):
            LoadedDocument(b"")


class TestRender:
    def test_render_page_returns_pil_image(self):
        b = _make_simple_pdf()
        with load(b) as doc:
            img = doc.render_page(0, dpi=100)
            assert img.mode == "RGB"
            # Letter at 100 DPI: ~850 x 1100
            assert img.width > 500
            assert img.height > 700


class TestText:
    def test_all_text_contains_content(self):
        b = _make_simple_pdf("HELLO_TAG")
        with load(b) as doc:
            assert "HELLO_TAG" in doc.all_text()

    def test_text_spans_have_font_info(self):
        b = _make_simple_pdf("with fonts")
        with load(b) as doc:
            spans = doc.text_spans(0)
            assert spans, "expected at least one span"
            assert any(s.font for s in spans)
            assert all(s.size > 0 for s in spans)

    def test_text_lines_grouped(self):
        b = _make_simple_pdf("group me")
        with load(b) as doc:
            lines = doc.text_lines(0)
            assert lines, "expected at least one line"
            assert all(len(line) >= 1 for line in lines)


class TestMetadata:
    def test_info_dict_has_producer_when_present(self):
        b = _make_simple_pdf()
        with load(b) as doc:
            info = doc.info_dict()
            # reportlab sets /Producer
            assert info.producer is not None
            assert "ReportLab" in (info.producer or "") or info.raw

    def test_raw_dict_populated(self):
        b = _make_simple_pdf()
        with load(b) as doc:
            info = doc.info_dict()
            # There should be at least one entry (Producer)
            assert any(k.startswith("/") for k in info.raw)


class TestXref:
    def test_clean_pdf_has_one_eof(self):
        b = _make_simple_pdf()
        with load(b) as doc:
            summary = doc.xref_summary()
            assert summary.num_eof_markers >= 1
            assert not summary.has_incremental_updates

    def test_incremental_update_detected(self):
        b0 = _make_simple_pdf("v1")
        b1 = _apply_incremental_update(b0)
        with load(b1) as doc:
            summary = doc.xref_summary()
            assert summary.num_eof_markers >= 2
            assert summary.has_incremental_updates


class TestFonts:
    def test_fonts_reported(self):
        b = _make_simple_pdf()
        with load(b) as doc:
            fonts = doc.fonts()
            assert fonts, "expected at least one font entry"
            # reportlab uses the built-in Helvetica; pymupdf may report the
            # basefont name in either the .name field. Accept anything with a
            # non-empty name.
            assert all(f.name for f in fonts)


class TestAsDict:
    def test_summary_shape(self):
        b = _make_simple_pdf()
        with load(b) as doc:
            d = doc.as_dict()
            assert d["num_pages"] == 1
            assert d["document_id"].startswith("sha256:")
            assert "producer" in d
            assert "has_incremental_updates" in d
