"""Smoke tests for the test corpus generator.

Regenerates the corpus in a temp directory (via the underlying builders +
tampering ops) and asserts the produced PDFs load cleanly and expose the
signatures we expect the detectors to key on. This is NOT the integration
harness — that lives in tests/integration/. This is just "did we build
plausible PDFs at all."
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tamper_detect.loader import load
from testdata.pdf_builders import (
    build_ein_letter,
    build_sales_tax_permit,
    build_utility_invoice,
)
from testdata.tamper_ops import (
    ai_lookalike,
    copy_move_logo,
    edit_amount_incremental,
    overlay_swap_text,
    resave_as_word,
    scan_and_splice_amount,
    swap_ein_incremental,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCleanBuilders:
    def test_utility_invoice_has_expected_fields(self):
        with load(build_utility_invoice()) as doc:
            text = doc.all_text()
            assert "Metro Water" in text
            assert "$87.42" in text
            assert doc.num_pages == 1

    def test_permit_has_expected_fields(self):
        with load(build_sales_tax_permit()) as doc:
            text = doc.all_text()
            assert "SALES TAX PERMIT" in text
            assert "Blue Ocean Coffee LLC" in text

    def test_ein_has_expected_fields(self):
        with load(build_ein_letter()) as doc:
            text = doc.all_text()
            assert "88-3341572" in text
            assert "BLUE OCEAN COFFEE LLC" in text

    def test_clean_docs_have_single_eof(self):
        for pdf in (build_utility_invoice(), build_sales_tax_permit(), build_ein_letter()):
            with load(pdf) as doc:
                assert doc.xref_summary().num_eof_markers >= 1
                assert not doc.xref_summary().has_incremental_updates


class TestTamperOps:
    def test_edit_amount_incremental_triggers_incremental(self):
        clean = build_utility_invoice()
        tampered = edit_amount_incremental(clean, "$87.42", "$874.20")
        with load(tampered) as doc:
            assert doc.xref_summary().has_incremental_updates
            # Text layer still has the old amount somewhere.
            assert "$87.42" in doc.all_text()

    def test_scan_and_splice_produces_larger_pdf(self):
        clean = build_utility_invoice()
        tampered = scan_and_splice_amount(clean, "$87.42", "$874.20")
        # Image-wrapped PDF is much larger than the clean text-only one.
        assert len(tampered) > 20 * len(clean)
        with load(tampered) as doc:
            # There should now be an embedded image on page 1.
            imgs = doc.embedded_images(0)
            assert len(imgs) >= 1

    def test_overlay_swap_preserves_old_text_in_layer(self):
        clean = build_sales_tax_permit()
        tampered = overlay_swap_text(clean, "Blue Ocean Coffee LLC", "Red Mountain Tea LLC")
        with load(tampered) as doc:
            text = doc.all_text()
            # Old text remains in the text stream even though rendered pixels are covered.
            assert "Blue Ocean Coffee LLC" in text
            assert "Red Mountain Tea LLC" in text

    def test_resave_as_word_sets_mismatched_producer_creator(self):
        clean = build_sales_tax_permit()
        tampered = resave_as_word(clean)
        with load(tampered) as doc:
            info = doc.info_dict()
            assert info.creator and "Word" in info.creator
            assert info.producer and "iLovePDF" in info.producer

    def test_swap_ein_incremental_triggers_incremental_and_mixes_fonts(self):
        clean = build_ein_letter()
        tampered = swap_ein_incremental(clean, "88-3341572", "77-9998880")
        with load(tampered) as doc:
            assert doc.xref_summary().has_incremental_updates
            # The overlay uses Times; original uses Courier. Distinct fonts should exist.
            font_names = {f.name for f in doc.fonts()}
            assert len(font_names) >= 2

    def test_ai_lookalike_is_image_only(self):
        clean = build_ein_letter()
        tampered = ai_lookalike(clean)
        with load(tampered) as doc:
            # No text layer content of substance
            assert len(doc.all_text().strip()) == 0
            # But there is at least one embedded image
            assert len(doc.embedded_images(0)) >= 1

    def test_copy_move_produces_image_pdf(self):
        clean = build_utility_invoice()
        tampered = copy_move_logo(clean)
        with load(tampered) as doc:
            assert len(doc.embedded_images(0)) >= 1


class TestGenerator:
    def test_generate_writes_labels_json(self, tmp_path, monkeypatch):
        # Import inside the test so we can patch the paths.
        from testdata import generate as gen

        monkeypatch.setattr(gen, "CLEAN_DIR", tmp_path / "clean")
        monkeypatch.setattr(gen, "TAMPERED_DIR", tmp_path / "tampered")
        monkeypatch.setattr(gen, "LABELS_PATH", tmp_path / "labels.json")
        # generator uses REPO_ROOT to compute relative paths; point it at tmp_path
        monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)

        gen.main()

        data = json.loads((tmp_path / "labels.json").read_text())
        assert len(data["docs"]) == 10
        assert sum(1 for d in data["docs"] if d["authentic"]) == 3
        assert sum(1 for d in data["docs"] if not d["authentic"]) == 7


class TestExistingCorpusOnDisk:
    """These run against the corpus already written to testdata/."""

    def setup_method(self):
        labels_path = REPO_ROOT / "testdata" / "labels.json"
        if not labels_path.exists():
            pytest.skip("corpus not generated yet — run: python -m testdata.generate")
        self.labels = json.loads(labels_path.read_text())

    def test_all_files_exist(self):
        for entry in self.labels["docs"]:
            p = REPO_ROOT / entry["file"]
            assert p.exists(), f"missing corpus file: {p}"

    def test_all_files_load(self):
        for entry in self.labels["docs"]:
            p = REPO_ROOT / entry["file"]
            with load(p.read_bytes()) as doc:
                assert doc.num_pages >= 1
