"""tamper-detect CLI — analyze a single PDF and print the report."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tamper_detect.analyze import analyze


app = typer.Typer(add_completion=False, help="Tamper-detection layer for KYC PDF documents.")
_console = Console()


@app.command()
def main(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="Path to the PDF to analyze."),
    ocr_text_file: Path | None = typer.Option(
        None, "--ocr-text-file", help="Optional file with the reference OCR text supplied by the caller."
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the raw report JSON."),
    no_narrative: bool = typer.Option(False, "--no-narrative", help="Skip the Claude narrative."),
) -> None:
    """Analyze PDF and print a report."""
    supplied = ocr_text_file.read_text() if ocr_text_file else None
    enable_narrative = None if not no_narrative else False
    report = analyze(pdf.read_bytes(), supplied_ocr_text=supplied, enable_narrative=enable_narrative)

    if json_out:
        _console.print_json(report.model_dump_json())
        return

    color = {"pass": "green", "review": "yellow", "fail": "red"}[report.decision]
    _console.print(
        f"[bold]Document:[/] {pdf.name}   "
        f"[bold]Type:[/] {report.doc_type_hint.value}   "
        f"[bold]Score:[/] {report.overall_score:.2f}   "
        f"[bold {color}]{report.decision.upper()}[/]"
    )

    if report.findings:
        table = Table(title="Findings", show_lines=False)
        table.add_column("Signal")
        table.add_column("Tier")
        table.add_column("Score", justify="right")
        table.add_column("Weight", justify="right")
        table.add_column("Evidence")
        for f in report.findings:
            ev_summary = _summarize_evidence(f.evidence)
            table.add_row(
                f.signal,
                f.tier,
                f"{f.score:.2f}",
                f"{f.weight_applied:.2f}" if f.weight_applied is not None else "-",
                ev_summary,
            )
        _console.print(table)
    else:
        _console.print("[dim]No findings.[/]")

    if report.narrative:
        _console.print()
        _console.print("[bold]Narrative:[/]")
        _console.print(report.narrative)

    _console.print(f"[dim]{len(report.meta.detectors_run)} detectors ran in {report.meta.runtime_ms} ms[/]")


def _summarize_evidence(ev: dict, max_len: int = 90) -> str:
    text = json.dumps(ev, default=str)
    return text if len(text) <= max_len else text[:max_len] + "..."


if __name__ == "__main__":  # pragma: no cover
    app()
