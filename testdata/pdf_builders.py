"""Clean-document builders for the test corpus.

Uses reportlab to synthesize plausible-looking utility invoices, sales-tax
permits, and IRS EIN letters. These are the "authentic" ground-truth PDFs.
Tampered variants are built by applying operations in `tamper_ops.py` on top.
"""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


# Page dimensions.
_PAGE_W, _PAGE_H = LETTER


def _draw_faux_logo(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str) -> None:
    """A tiny placeholder logo (colored rect + initials) so we have raster-ish
    elements to inspect."""
    c.saveState()
    c.setFillColor(colors.HexColor("#1f4e79"))
    c.rect(x, y, w, h, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(x + w / 2, y + h / 2 - 4, label)
    c.restoreState()


# ---------------------------------------------------------------------------
# Utility invoice
# ---------------------------------------------------------------------------


def build_utility_invoice(
    *,
    company: str = "Metro Water & Power",
    account_number: str = "0421-889-3311",
    customer_name: str = "Blue Ocean Coffee LLC",
    service_address: str = "1420 Rivera Ave, Portland, OR 97201",
    bill_date: str = "2026-06-14",
    due_date: str = "2026-07-08",
    amount_due: str = "$87.42",
) -> bytes:
    """Return the PDF bytes for a plausible utility bill."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.setTitle("Utility Statement")
    c.setAuthor("Metro Water & Power")
    c.setCreator("Metro Water & Power Billing System v2.4")
    c.setProducer("Metro Water & Power Billing System v2.4")

    # Header
    _draw_faux_logo(c, 72, _PAGE_H - 108, 60, 36, "MWP")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(144, _PAGE_H - 92, company)
    c.setFont("Helvetica", 9)
    c.drawString(144, _PAGE_H - 106, "Customer Service: 1-800-555-0142")

    # Account block
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, _PAGE_H - 160, "Account Statement")
    c.setFont("Helvetica", 10)
    y = _PAGE_H - 180
    for label, value in [
        ("Account Number", account_number),
        ("Customer Name", customer_name),
        ("Service Address", service_address),
        ("Bill Date", bill_date),
        ("Due Date", due_date),
    ]:
        c.drawString(72, y, f"{label}:")
        c.drawString(200, y, value)
        y -= 14

    # Charges box
    y -= 12
    c.setFillColor(colors.HexColor("#f2f2f2"))
    c.rect(72, y - 60, _PAGE_W - 144, 60, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(84, y - 18, "Current Charges")
    c.setFont("Helvetica", 10)
    c.drawString(84, y - 34, "Water Usage (2,140 gal)")
    c.drawRightString(_PAGE_W - 84, y - 34, "$61.20")
    c.drawString(84, y - 48, "Service Fee")
    c.drawRightString(_PAGE_W - 84, y - 48, "$26.22")

    # Total
    y_total = y - 90
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, y_total, "TOTAL DUE:")
    c.drawRightString(_PAGE_W - 84, y_total, amount_due)

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(72, 54, "Please pay by the due date to avoid a late fee.")
    c.drawString(72, 42, "This is a computer-generated statement.")

    c.showPage()
    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sales tax permit
# ---------------------------------------------------------------------------


def build_sales_tax_permit(
    *,
    state: str = "OREGON",
    permit_number: str = "STP-2026-118-4471",
    business_name: str = "Blue Ocean Coffee LLC",
    dba: str = "Blue Ocean Coffee",
    address: str = "1420 Rivera Ave, Portland, OR 97201",
    effective_date: str = "2026-04-01",
    expiration_date: str = "2027-03-31",
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.setTitle("Sales Tax Permit")
    c.setAuthor(f"State of {state.title()} Department of Revenue")
    c.setCreator("DOR Permit Issuance System")
    c.setProducer("DOR Permit Issuance System")

    # State seal placeholder
    _draw_faux_logo(c, _PAGE_W / 2 - 30, _PAGE_H - 108, 60, 60, state[:2])

    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(_PAGE_W / 2, _PAGE_H - 138, f"STATE OF {state}")
    c.setFont("Helvetica", 12)
    c.drawCentredString(_PAGE_W / 2, _PAGE_H - 156, "Department of Revenue")
    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(_PAGE_W / 2, _PAGE_H - 190, "SALES TAX PERMIT")

    # Body
    c.setFont("Helvetica", 11)
    y = _PAGE_H - 240
    for label, value in [
        ("Permit Number:", permit_number),
        ("Business Name:", business_name),
        ("Doing Business As:", dba),
        ("Business Address:", address),
        ("Effective Date:", effective_date),
        ("Expiration Date:", expiration_date),
    ]:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(120, y, label)
        c.setFont("Helvetica", 11)
        c.drawString(280, y, value)
        y -= 22

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        120,
        y - 30,
        "This permit authorizes the holder to collect sales tax on taxable transactions.",
    )
    c.drawString(120, y - 44, "Display prominently at your place of business.")

    c.showPage()
    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# EIN letter (IRS CP 575 look-alike)
# ---------------------------------------------------------------------------


def build_ein_letter(
    *,
    ein: str = "88-3341572",
    business_name: str = "BLUE OCEAN COFFEE LLC",
    address_line_1: str = "1420 RIVERA AVE",
    address_line_2: str = "PORTLAND OR 97201",
    letter_date: str = "MAY 12, 2026",
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    c.setTitle("EIN Assignment Notice")
    c.setAuthor("Internal Revenue Service")
    c.setCreator("IRS EIN Online Assistant")
    c.setProducer("IRS EIN Online Assistant")

    # Header
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, _PAGE_H - 90, "IRS")
    c.setFont("Helvetica", 10)
    c.drawString(96, _PAGE_H - 90, "Department of the Treasury")
    c.drawString(96, _PAGE_H - 104, "Internal Revenue Service")
    c.drawRightString(_PAGE_W - 72, _PAGE_H - 90, f"Date of this notice:  {letter_date}")
    c.drawRightString(_PAGE_W - 72, _PAGE_H - 104, "Employer Identification Number:")

    # EIN (monospace, prominent)
    c.setFont("Courier-Bold", 14)
    c.drawRightString(_PAGE_W - 72, _PAGE_H - 122, ein)

    # Addressee block
    c.setFont("Helvetica", 11)
    c.drawString(72, _PAGE_H - 170, business_name)
    c.drawString(72, _PAGE_H - 186, address_line_1)
    c.drawString(72, _PAGE_H - 202, address_line_2)

    # Body
    c.setFont("Helvetica", 10)
    body = [
        "Thank you for applying for an Employer Identification Number (EIN).",
        "We assigned you EIN " + ein + ". This EIN will identify your business",
        "account and tax returns. Please keep this notice in your permanent records.",
        "",
        "When filing tax documents, making payments, or replying to any related",
        "correspondence, it is very important that you use your EIN and complete",
        "name and address exactly as shown above.",
    ]
    y = _PAGE_H - 250
    for line in body:
        c.drawString(72, y, line)
        y -= 14

    c.setFont("Helvetica-Bold", 10)
    c.drawString(72, y - 10, "Keep this notice for your records.")

    c.showPage()
    c.save()
    return buf.getvalue()
