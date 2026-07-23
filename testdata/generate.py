"""Corpus generator — produces the 10 test PDFs + labels.json.

Run: python -m testdata.generate

Idempotent — safe to rerun; overwrites existing files under testdata/clean and
testdata/tampered.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = REPO_ROOT / "testdata" / "clean"
TAMPERED_DIR = REPO_ROOT / "testdata" / "tampered"
LABELS_PATH = REPO_ROOT / "testdata" / "labels.json"


@dataclass
class CorpusEntry:
    file: str          # relative to repo root
    doc_type: str
    authentic: bool
    tamper_kind: str | None
    expected_signals: list[str]
    expected_decision: str  # "pass" | "review_or_fail"
    original_text_reference: str | None = None
    notes: str = ""


@dataclass
class Corpus:
    docs: list[CorpusEntry] = field(default_factory=list)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def build_corpus() -> Corpus:
    corpus = Corpus()

    # ---- 1. Utility invoice — clean --------------------------------------
    clean_bill = build_utility_invoice()
    p = CLEAN_DIR / "01_utility_invoice_clean.pdf"
    _write(p, clean_bill)
    corpus.docs.append(
        CorpusEntry(
            file=str(p.relative_to(REPO_ROOT)),
            doc_type="utility_invoice",
            authentic=True,
            tamper_kind=None,
            expected_signals=[],
            expected_decision="pass",
            original_text_reference=None,
        )
    )

    # ---- 2. Utility invoice — amount edited via incremental update -------
    tampered_2 = edit_amount_incremental(clean_bill, "$87.42", "$874.20")
    p = TAMPERED_DIR / "02_utility_invoice_amount_incremental.pdf"
    _write(p, tampered_2)
    corpus.docs.append(
        CorpusEntry(
            file=str(p.relative_to(REPO_ROOT)),
            doc_type="utility_invoice",
            authentic=False,
            tamper_kind="amount_edit_incremental_update",
            expected_signals=[
                "incremental_update_present",
                "text_layer_mismatches_render",
            ],
            expected_decision="review_or_fail",
            original_text_reference="$87.42",
            notes="Text layer still says $87.42; new $874.20 stamped on top.",
        )
    )

    # ---- 3. Utility invoice — scanned + spliced amount -------------------
    tampered_3 = scan_and_splice_amount(clean_bill, "$87.42", "$874.20")
    p = TAMPERED_DIR / "03_utility_invoice_scanned_spliced.pdf"
    _write(p, tampered_3)
    corpus.docs.append(
        CorpusEntry(
            file=str(p.relative_to(REPO_ROOT)),
            doc_type="utility_invoice",
            authentic=False,
            tamper_kind="scanned_amount_splice",
            expected_signals=[
                "jpeg_ghost",
                "ela_hotspots",
                "text_layer_mismatches_render",
            ],
            expected_decision="review_or_fail",
            original_text_reference="$87.42",
            notes="Whole page rasterized at QF=88; amount region re-saved at QF=62.",
        )
    )

    # ---- 4. Sales tax permit — clean -------------------------------------
    clean_permit = build_sales_tax_permit()
    p = CLEAN_DIR / "04_sales_tax_permit_clean.pdf"
    _write(p, clean_permit)
    corpus.docs.append(
        CorpusEntry(
            file=str(p.relative_to(REPO_ROOT)),
            doc_type="sales_tax_permit",
            authentic=True,
            tamper_kind=None,
            expected_signals=[],
            expected_decision="pass",
        )
    )

    # ---- 5. Sales tax permit — business name swapped via overlay ---------
    tampered_5 = overlay_swap_text(
        clean_permit,
        "Blue Ocean Coffee LLC",
        "Red Mountain Tea LLC",
    )
    p = TAMPERED_DIR / "05_sales_tax_permit_overlay_swap.pdf"
    _write(p, tampered_5)
    corpus.docs.append(
        CorpusEntry(
            file=str(p.relative_to(REPO_ROOT)),
            doc_type="sales_tax_permit",
            authentic=False,
            tamper_kind="business_name_overlay",
            expected_signals=[
                "text_over_image_overlay",
                "text_layer_mismatches_render",
            ],
            expected_decision="review_or_fail",
            original_text_reference="Blue Ocean Coffee LLC",
            notes="Original text NOT removed; white rect + new text stamped on top.",
        )
    )

    # ---- 6. Sales tax permit — re-saved through Word-like path -----------
    tampered_6 = resave_as_word(clean_permit)
    p = TAMPERED_DIR / "06_sales_tax_permit_resave_word.pdf"
    _write(p, tampered_6)
    corpus.docs.append(
        CorpusEntry(
            file=str(p.relative_to(REPO_ROOT)),
            doc_type="sales_tax_permit",
            authentic=False,
            tamper_kind="resave_word_producer_mismatch",
            expected_signals=["producer_creator_mismatch"],
            expected_decision="review_or_fail",
            notes="Creator=Microsoft Word, Producer=iLovePDF.",
        )
    )

    # ---- 7. EIN letter — clean ------------------------------------------
    clean_ein = build_ein_letter()
    p = CLEAN_DIR / "07_ein_letter_clean.pdf"
    _write(p, clean_ein)
    corpus.docs.append(
        CorpusEntry(
            file=str(p.relative_to(REPO_ROOT)),
            doc_type="ein_letter",
            authentic=True,
            tamper_kind=None,
            expected_signals=[],
            expected_decision="pass",
        )
    )

    # ---- 8. EIN letter — EIN digits swapped via mixed font ---------------
    tampered_8 = swap_ein_incremental(clean_ein, "88-3341572", "77-9998880")
    p = TAMPERED_DIR / "08_ein_letter_ein_swap_mixed_font.pdf"
    _write(p, tampered_8)
    corpus.docs.append(
        CorpusEntry(
            file=str(p.relative_to(REPO_ROOT)),
            doc_type="ein_letter",
            authentic=False,
            tamper_kind="ein_swap_mixed_font_incremental",
            expected_signals=[
                "incremental_update_present",
                "font_family_mix_in_line",
                "text_layer_mismatches_render",
            ],
            expected_decision="review_or_fail",
            original_text_reference="88-3341572",
            notes="Original EIN in Courier; overlay in Times; incremental save.",
        )
    )

    # ---- 9. EIN letter — AI-generated look-alike -------------------------
    tampered_9 = ai_lookalike(clean_ein)
    p = TAMPERED_DIR / "09_ein_letter_ai_lookalike.pdf"
    _write(p, tampered_9)
    corpus.docs.append(
        CorpusEntry(
            file=str(p.relative_to(REPO_ROOT)),
            doc_type="ein_letter",
            authentic=False,
            tamper_kind="ai_generated_lookalike",
            expected_signals=[
                "noise_residual_inconsistency",
                "ai_generated_image",
                "template_mismatch",
            ],
            expected_decision="review_or_fail",
            notes="Whole page rasterized w/ Gaussian noise + low-QF JPEG.",
        )
    )

    # ---- 10. Utility invoice — copy-moved logo/stamp ---------------------
    tampered_10 = copy_move_logo(clean_bill)
    p = TAMPERED_DIR / "10_utility_invoice_copy_move.pdf"
    _write(p, tampered_10)
    corpus.docs.append(
        CorpusEntry(
            file=str(p.relative_to(REPO_ROOT)),
            doc_type="utility_invoice",
            authentic=False,
            tamper_kind="copy_move_cover_watermark",
            expected_signals=["copy_move", "ela_hotspots"],
            expected_decision="review_or_fail",
            notes="A blank region is duplicated to cover an underlying PAID watermark.",
        )
    )

    return corpus


def write_labels(corpus: Corpus) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(
        json.dumps({"docs": [asdict(d) for d in corpus.docs]}, indent=2) + "\n"
    )


def main() -> None:
    corpus = build_corpus()
    write_labels(corpus)
    print(f"Wrote {len(corpus.docs)} documents.")
    print(f"Labels: {LABELS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
