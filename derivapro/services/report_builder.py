"""Model validation report generation pipeline.

Collects pricing results, sensitivity/scenario/convergence analysis output,
AI assessments, and model governance text into a single ``ReportTemplate``
dataclass, then renders that structured data into a PDF document with
ReportLab's Platypus layout engine.

The same ``ReportTemplate`` instance can also be handed to the HTML preview
(``report_template.html``) via ``dataclasses.asdict``, so the PDF and the
in-browser preview are always built from identical data.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_STYLES = getSampleStyleSheet()
_TITLE_STYLE = ParagraphStyle(
    "ReportTitle", parent=_STYLES["Title"], fontSize=20, spaceAfter=6
)
_SUBTITLE_STYLE = ParagraphStyle(
    "ReportSubtitle",
    parent=_STYLES["Normal"],
    fontSize=12,
    textColor=colors.grey,
    spaceAfter=18,
)
_HEADING_STYLE = ParagraphStyle(
    "SectionHeading", parent=_STYLES["Heading2"], spaceBefore=14, spaceAfter=6
)
_SUB_HEADING_STYLE = ParagraphStyle(
    "SubHeading", parent=_STYLES["Heading3"], spaceBefore=10, spaceAfter=4
)
_BODY_STYLE = ParagraphStyle("Body", parent=_STYLES["Normal"], spaceAfter=8, leading=14)


@dataclass
class ReportTemplate:
    """Structured content for a model validation report."""

    model_name: str = "Vanilla Option Model"
    validation_type: str = "Baseline Validation"
    validation_date: str = "2024-09-01"

    validation_overview: str = (
        "This validation provides an overview of the model performance and governance."
    )
    model_purpose: str = "The model is designed to price vanilla and exotic options."
    model_overview: str = (
        "This section provides an overview of the model's pricing mechanisms."
    )
    key_limitations: str = (
        "Key limitations include volatility assumptions and model calibration."
    )
    validation_scope: str = (
        "The validation scope includes conceptual soundness, model performance, "
        "and ongoing monitoring."
    )
    data_quality: str = "The data quality was assessed based on completeness, accuracy, and reliability."
    conceptual_soundness: str = (
        "The model is conceptually sound with a well-structured pricing mechanism."
    )
    scenario_description: str = (
        "Scenario analysis was conducted with stress on spot price, volatility, "
        "and interest rates."
    )
    benchmarking_description: str = (
        "Benchmarking analysis compares the model outputs with standard benchmarks."
    )
    rpbl_analysis: str = (
        "Risk P&L analysis compares realized profit and loss against theoretical "
        "risk-based estimates."
    )
    model_governance: str = "Governance details for the model."
    appendix_content: str = "Appendix content."

    sensitivity_description: str = ""
    sensitivity_combined: list = field(default_factory=list)
    sensitivity_plot: Optional[str] = None
    sensitivity_assessment: str = "No assessment available."

    scenario_results: list = field(default_factory=list)
    scenario_assessment: str = "No assessment available."

    convergence_plot: Optional[str] = None
    convergence_summary: list = field(default_factory=list)
    convergence_assessment: str = "No assessment available."

    latest_pricing_result: Any = None
    latest_analysis: Any = None


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def _styled_table(data: list) -> Table:
    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    return table


def _add_plot(story: list, plot_filename: Optional[str], static_dir: str) -> None:
    if not plot_filename:
        story.append(Paragraph("No plot available.", _BODY_STYLE))
        return

    plot_path = os.path.join(static_dir, plot_filename)
    if not os.path.isfile(plot_path):
        story.append(Paragraph("Plot file not found on disk.", _BODY_STYLE))
        return

    try:
        story.append(Image(plot_path, width=5.5 * inch, height=3.3 * inch))
        story.append(Spacer(1, 8))
    except Exception:
        story.append(Paragraph("Unable to embed plot image.", _BODY_STYLE))


def render_report_pdf(report: ReportTemplate, static_dir: str) -> bytes:
    """Render a ``ReportTemplate`` into PDF bytes using ReportLab Platypus.

    ``static_dir`` is the on-disk directory holding the plot images
    referenced by ``report.sensitivity_plot`` / ``report.convergence_plot``
    so they can be embedded in the document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title=f"{report.model_name} Validation Report",
    )

    story: list = []
    story.append(Paragraph(f"{report.model_name} Validation Report", _TITLE_STYLE))
    story.append(
        Paragraph(
            f"{report.validation_type} &mdash; Date: {report.validation_date}",
            _SUBTITLE_STYLE,
        )
    )

    def add_section(number: str, heading: str, text: Optional[str]) -> None:
        story.append(Paragraph(f"{number}. {heading}", _HEADING_STYLE))
        story.append(Paragraph(text or "N/A", _BODY_STYLE))

    add_section("1", "Executive Summary", report.validation_overview)
    add_section("2", "Model Purpose and Use", report.model_purpose)
    add_section("3", "Model Overview", report.model_overview)
    add_section(
        "4",
        "Key Limitations/Weaknesses and Associated Model Risks",
        report.key_limitations,
    )
    add_section("5", "Validation Scope and Approach", report.validation_scope)
    add_section("6", "Data Quality", report.data_quality)
    add_section("7", "Conceptual Soundness", report.conceptual_soundness)

    story.append(Paragraph("8. Model Performance", _HEADING_STYLE))

    if report.latest_pricing_result is not None:
        pr = report.latest_pricing_result
        story.append(Paragraph("8.1 Latest Pricing Result", _SUB_HEADING_STYLE))
        data = [["Metric", "Value"]]
        for label, value in (
            ("Price", getattr(pr, "price", None)),
            ("Delta", getattr(pr, "delta", None)),
            ("Gamma", getattr(pr, "gamma", None)),
            ("Vega", getattr(pr, "vega", None)),
            ("Theta", getattr(pr, "theta", None)),
            ("Rho", getattr(pr, "rho", None)),
        ):
            data.append([label, _fmt(value)])
        story.append(_styled_table(data))
        story.append(Spacer(1, 10))

    story.append(Paragraph("8.2 Sensitivity Analysis", _SUB_HEADING_STYLE))
    story.append(
        Paragraph(
            report.sensitivity_description or "No sensitivity analysis was performed.",
            _BODY_STYLE,
        )
    )
    _add_plot(story, report.sensitivity_plot, static_dir)
    story.append(
        Paragraph(
            "<b>MRM Assessment:</b> "
            + (report.sensitivity_assessment or "No assessment available."),
            _BODY_STYLE,
        )
    )

    story.append(Paragraph("8.3 Scenario Analysis", _SUB_HEADING_STYLE))
    story.append(Paragraph(report.scenario_description or "", _BODY_STYLE))
    if report.scenario_results:
        rows = [["Scenario", "Option Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]]
        for row in report.scenario_results:
            rows.append([
                row.get("scenario", ""),
                _fmt(row.get("option_price")),
                _fmt(row.get("delta")),
                _fmt(row.get("gamma")),
                _fmt(row.get("vega")),
                _fmt(row.get("theta")),
                _fmt(row.get("rho")),
            ])
        story.append(_styled_table(rows))
    else:
        story.append(Paragraph("No scenario results available.", _BODY_STYLE))
    story.append(
        Paragraph(
            "<b>MRM Assessment:</b> "
            + (report.scenario_assessment or "No assessment available."),
            _BODY_STYLE,
        )
    )

    story.append(Paragraph("8.4 Convergence Analysis", _SUB_HEADING_STYLE))
    _add_plot(story, report.convergence_plot, static_dir)
    if report.convergence_summary:
        rows = [["Step / Paths", "Value"]]
        for item in report.convergence_summary:
            try:
                rows.append([str(item[0]), _fmt(item[1])])
            except (TypeError, IndexError):
                rows.append([str(item), ""])
        story.append(_styled_table(rows))
    else:
        story.append(Paragraph("No convergence results available.", _BODY_STYLE))
    story.append(
        Paragraph(
            "<b>MRM Assessment:</b> "
            + (report.convergence_assessment or "No assessment available."),
            _BODY_STYLE,
        )
    )

    story.append(PageBreak())
    add_section("9", "Benchmarking", report.benchmarking_description)
    add_section("10", "Risk P&L (RPBL) Analysis", report.rpbl_analysis)
    add_section("11", "Model Governance", report.model_governance)
    add_section("12", "Appendix", report.appendix_content)

    doc.build(story)
    return buffer.getvalue()
