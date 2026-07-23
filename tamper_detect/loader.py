"""DocumentLoader — parse a PDF once, expose everything the detectors need.

Uses pymupdf (fitz) for page render, text spans with font info, and embedded
image extraction. Uses pikepdf for the low-level PDF object model — /Info dict,
xref inspection, and detecting incremental updates.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

import fitz  # pymupdf
import pikepdf
from PIL import Image


# Default render DPI for the OCR-vs-render and pixel-forensics passes.
DEFAULT_RENDER_DPI = 200


@dataclass(frozen=True)
class TextSpan:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    font: str
    size: float
    flags: int  # pymupdf font flags (bold, italic, etc.)


@dataclass(frozen=True)
class EmbeddedImage:
    page: int
    xref: int
    bbox: tuple[float, float, float, float] | None
    image: Image.Image
    ext: str  # "jpeg", "png", etc.


@dataclass(frozen=True)
class FontInfo:
    xref: int
    name: str          # local resource name (e.g. "F1")
    basefont: str      # real font name (e.g. "Helvetica-Bold")
    type: str          # "TrueType", "Type1", "Type0", ...
    embedded: bool
    subset: bool


@dataclass
class InfoDict:
    """Values from the PDF /Info dictionary. Everything is optional."""

    producer: str | None = None
    creator: str | None = None
    author: str | None = None
    title: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creation_date: str | None = None
    mod_date: str | None = None
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class XrefSummary:
    num_eof_markers: int
    has_incremental_updates: bool
    is_linearized: bool
    num_objects: int


class LoadedDocument:
    """A parsed PDF exposing lazy views over pages, images, metadata, and fonts.

    Owns two backing handles (pymupdf + pikepdf). Use as a context manager so
    both get closed cleanly.
    """

    def __init__(self, pdf_bytes: bytes) -> None:
        if not pdf_bytes:
            raise ValueError("pdf_bytes is empty")
        self.pdf_bytes = pdf_bytes
        self.document_id = "sha256:" + hashlib.sha256(pdf_bytes).hexdigest()
        self._fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        self._pikepdf_doc = pikepdf.open(io.BytesIO(pdf_bytes))
        self.num_pages = self._fitz_doc.page_count

    # --- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        try:
            self._fitz_doc.close()
        finally:
            self._pikepdf_doc.close()

    def __enter__(self) -> "LoadedDocument":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- render ------------------------------------------------------------

    def render_page(self, page_num: int, dpi: int = DEFAULT_RENDER_DPI) -> Image.Image:
        """Render a page to a PIL image at the given DPI."""
        page = self._fitz_doc[page_num]
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # --- text spans --------------------------------------------------------

    def text_spans(self, page_num: int) -> list[TextSpan]:
        """Return every text span on a page with its font info and bbox."""
        page = self._fitz_doc[page_num]
        raw = page.get_text("dict")
        spans: list[TextSpan] = []
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    spans.append(
                        TextSpan(
                            page=page_num,
                            bbox=tuple(span["bbox"]),  # type: ignore[arg-type]
                            text=span.get("text", ""),
                            font=span.get("font", ""),
                            size=float(span.get("size", 0.0)),
                            flags=int(span.get("flags", 0)),
                        )
                    )
        return spans

    def text_lines(self, page_num: int) -> list[list[TextSpan]]:
        """Group spans by line — helpful for font-mix-in-line checks."""
        page = self._fitz_doc[page_num]
        raw = page.get_text("dict")
        lines: list[list[TextSpan]] = []
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                line_spans: list[TextSpan] = []
                for span in line.get("spans", []):
                    line_spans.append(
                        TextSpan(
                            page=page_num,
                            bbox=tuple(span["bbox"]),  # type: ignore[arg-type]
                            text=span.get("text", ""),
                            font=span.get("font", ""),
                            size=float(span.get("size", 0.0)),
                            flags=int(span.get("flags", 0)),
                        )
                    )
                if line_spans:
                    lines.append(line_spans)
        return lines

    def all_text(self) -> str:
        """Concatenated text across all pages, page-broken by \\n\\n."""
        parts = []
        for i in range(self.num_pages):
            parts.append(self._fitz_doc[i].get_text("text"))
        return "\n\n".join(parts).strip()

    # --- embedded images ---------------------------------------------------

    def embedded_images(self, page_num: int) -> list[EmbeddedImage]:
        """Extract every embedded raster image on a page."""
        page = self._fitz_doc[page_num]
        images: list[EmbeddedImage] = []
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                info = self._fitz_doc.extract_image(xref)
            except Exception:
                continue
            try:
                pil = Image.open(io.BytesIO(info["image"]))
                pil.load()
            except Exception:
                continue
            bbox: tuple[float, float, float, float] | None = None
            for rect in page.get_image_rects(xref):
                bbox = tuple(rect)  # type: ignore[assignment]
                break
            images.append(
                EmbeddedImage(
                    page=page_num,
                    xref=xref,
                    bbox=bbox,
                    image=pil,
                    ext=str(info.get("ext", "")),
                )
            )
        return images

    # --- metadata / info dict ---------------------------------------------

    def info_dict(self) -> InfoDict:
        """Read the /Info dictionary from the pikepdf handle."""
        info = InfoDict()
        docinfo = self._pikepdf_doc.docinfo
        if docinfo is None:
            return info
        raw: dict[str, str] = {}
        for key, value in dict(docinfo).items():
            key_s = str(key)
            val_s = str(value)
            raw[key_s] = val_s
        info.raw = raw
        info.producer = raw.get("/Producer")
        info.creator = raw.get("/Creator")
        info.author = raw.get("/Author")
        info.title = raw.get("/Title")
        info.subject = raw.get("/Subject")
        info.keywords = raw.get("/Keywords")
        info.creation_date = raw.get("/CreationDate")
        info.mod_date = raw.get("/ModDate")
        return info

    # --- xref + incremental updates ---------------------------------------

    def xref_summary(self) -> XrefSummary:
        """Structural summary useful for the incremental-update detector."""
        num_eof = self.pdf_bytes.count(b"%%EOF")
        # Any PDF has at least one %%EOF marker. More than one indicates
        # incremental updates (or, rarely, a hand-written PDF).
        has_updates = num_eof > 1
        is_linearized = getattr(self._pikepdf_doc, "is_linearized", False)
        try:
            num_objects = len(self._pikepdf_doc.objects)
        except Exception:
            num_objects = 0
        return XrefSummary(
            num_eof_markers=num_eof,
            has_incremental_updates=has_updates,
            is_linearized=is_linearized,
            num_objects=num_objects,
        )

    # --- fonts -------------------------------------------------------------

    def fonts(self) -> list[FontInfo]:
        """Aggregate font information across all pages."""
        seen: dict[int, FontInfo] = {}
        for page_num in range(self.num_pages):
            page = self._fitz_doc[page_num]
            for f in page.get_fonts(full=True):
                # (xref, ext, type, basefont, name, encoding, referencer)
                xref = int(f[0])
                if xref in seen:
                    continue
                ext = str(f[1])
                ftype = str(f[2])
                basefont = str(f[3]) if len(f) > 3 else ""
                name = str(f[4]) if len(f) > 4 else basefont
                embedded = ext.lower() != "n/a" and ext.lower() != "" and ext.lower() != "none"
                # A subset font typically has a "ABCDEF+" prefix on the basefont
                subset = "+" in basefont and basefont.split("+", 1)[0].isalpha()
                seen[xref] = FontInfo(
                    xref=xref,
                    name=name or basefont,
                    basefont=basefont,
                    type=ftype,
                    embedded=embedded,
                    subset=subset,
                )
        return list(seen.values())

    # --- convenience -------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        """Small summary useful for debugging."""
        info = self.info_dict()
        xref = self.xref_summary()
        return {
            "document_id": self.document_id,
            "num_pages": self.num_pages,
            "producer": info.producer,
            "creator": info.creator,
            "creation_date": info.creation_date,
            "mod_date": info.mod_date,
            "has_incremental_updates": xref.has_incremental_updates,
            "num_eof_markers": xref.num_eof_markers,
        }


def load(pdf_bytes: bytes) -> LoadedDocument:
    """Convenience factory."""
    return LoadedDocument(pdf_bytes)
