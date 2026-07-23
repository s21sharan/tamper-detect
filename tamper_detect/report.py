"""Report and Finding data models — the shape every detector, the scorer, and
the API surface agree on."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Tier = Literal["structural", "pixel", "ocr_diff", "experimental"]
Decision = Literal["pass", "review", "fail"]


class DocType(str, Enum):
    UTILITY_INVOICE = "utility_invoice"
    SALES_TAX_PERMIT = "sales_tax_permit"
    EIN_LETTER = "ein_letter"
    UNKNOWN = "unknown"


class Finding(BaseModel):
    """A single detector observation about a document."""

    model_config = ConfigDict(extra="forbid")

    signal: str = Field(..., description="Stable signal identifier, e.g. 'jpeg_ghost'")
    tier: Tier
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized 0..1; 1 = strong tamper")
    evidence: dict[str, Any] = Field(default_factory=dict)
    weight_applied: float | None = Field(
        default=None,
        description="tier_weight * signal_weight, filled in by the scorer",
    )

    @field_validator("signal")
    @classmethod
    def _signal_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("signal must be non-empty")
        return v


class ReportMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detectors_run: list[str] = Field(default_factory=list)
    runtime_ms: int = 0
    version: str = "0.1.0"


class Report(BaseModel):
    """The final tamper-detection report returned to the caller."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(..., description="sha256:<hex> of the input bytes")
    doc_type_hint: DocType = DocType.UNKNOWN
    overall_score: float = Field(..., ge=0.0, le=1.0)
    decision: Decision
    findings: list[Finding] = Field(default_factory=list)
    narrative: str | None = None
    meta: ReportMeta = Field(default_factory=ReportMeta)
