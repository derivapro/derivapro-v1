"""Product-agnostic AI assessment service.

A product page hands this module two things: the *input parameters* the user
submitted and the *model outputs* the pricing engine produced. The module turns
them into a structured prompt, sends it to the configured LLM provider (Azure /
Atlas by default, see ``derivapro.llm``) and returns a plain-text assessment.

Adding AI Assessment to another product is a two-step job:

1. Register a spec once at import time (usually next to the product's
   blueprint)::

       register_assessment_product(
           AssessmentProductSpec(
               key="my-product",                     # used in the URL
               title="My Product",                   # shown to the LLM and the user
               product_type="my_product",            # Instrument.product_type for persistence
               input_labels={"notional": "Notional", ...},
               focus="Comment on ...",               # optional product-specific guidance
           )
       )

2. Drop the macro onto the page (see ``templates/components/ai_assessment.html``)::

       {% import "components/ai_assessment.html" as ai %}
       {{ ai.panel("my-product", inputs=form_data, outputs=results) }}

The default input/output formatters understand the shared result shape used by
most pricing pipelines in DerivaPro (``primary_metrics``, ``scenarios``,
``benchmark_metrics``, ``cashflows``, ``summary``). Products with unusual output
payloads can supply ``input_formatter`` / ``output_formatter`` callables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..utils.lazy_imports import LazyAttribute

logger = logging.getLogger(__name__)

llm_client = LazyAttribute("derivapro.llm", "llm_client")

ANALYSIS_TYPE = "ai_assessment"

# Form keys that never belong in a prompt (framework plumbing, UI state).
DEFAULT_SKIP_INPUTS = frozenset(
    {
        "csrf_token",
        "analysis_type",
        "action",
        "benchmark_preset",
        "submit",
    }
)

# Result keys that are purely presentational (chart widths, colours) and add
# noise rather than information for the model.
DEFAULT_SKIP_OUTPUTS = frozenset({"analysis_visuals", "plot_filename", "plots"})

MAX_TABLE_ROWS = 24
MAX_SCHEDULE_ROWS = 40
MAX_PROMPT_CHARS = 24_000

InputFormatter = Callable[[Dict[str, Any]], List[Tuple[str, str]]]
OutputFormatter = Callable[[Dict[str, Any]], List["AssessmentSection"]]


@dataclass
class AssessmentSection:
    """A titled block of lines that becomes one section of the prompt."""

    title: str
    lines: List[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.lines:
            return ""
        body = "\n".join(self.lines)
        return f"{self.title}\n{body}"


@dataclass
class AssessmentProductSpec:
    """Everything the generic endpoint needs to know about one product."""

    key: str
    title: str
    product_type: Optional[str] = None
    # name -> human label; insertion order also drives the order inputs appear in the prompt
    input_labels: Dict[str, str] = field(default_factory=dict)
    # name -> {option_value: option_label} for select fields, so "2" reads as "Semiannual (2)"
    input_options: Dict[str, Dict[str, str]] = field(default_factory=dict)
    skip_inputs: frozenset = DEFAULT_SKIP_INPUTS
    skip_outputs: frozenset = DEFAULT_SKIP_OUTPUTS
    input_formatter: Optional[InputFormatter] = None
    output_formatter: Optional[OutputFormatter] = None
    focus: str = ""
    word_limit: int = 250


@dataclass
class AssessmentResult:
    ok: bool
    assessment: str
    model: Optional[str] = None
    provider: Optional[str] = None
    prompt_chars: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "assessment": self.assessment,
            "model": self.model,
            "provider": self.provider,
            "prompt_chars": self.prompt_chars,
            "error": self.error,
        }


_REGISTRY: Dict[str, AssessmentProductSpec] = {}


def register_assessment_product(spec: AssessmentProductSpec) -> AssessmentProductSpec:
    """Register (or replace) the assessment spec for ``spec.key``."""
    _REGISTRY[spec.key] = spec
    return spec


def get_assessment_product(key: str) -> Optional[AssessmentProductSpec]:
    return _REGISTRY.get(key)


def registered_assessment_products() -> Dict[str, AssessmentProductSpec]:
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def label_for(key: str, labels: Optional[Dict[str, str]] = None) -> str:
    if labels and key in labels:
        return labels[key]
    return str(key).replace("_", " ").strip().title()


def _iter_config_fields(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    sections = config.get("field_sections")
    if sections is None and "fields" in config:
        sections = [{"fields": config["fields"]}]
    for section in sections or []:
        for item in section.get("fields", []):
            if item.get("name"):
                yield item


def labels_from_field_sections(config: Dict[str, Any]) -> Dict[str, str]:
    """Collect ``name -> label`` (in form order) from the ``fields`` /
    ``field_sections`` config layout used by the fixed-income extension pages."""
    return {item["name"]: item.get("label", label_for(item["name"])) for item in _iter_config_fields(config)}


def options_from_field_sections(config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Collect ``name -> {value: label}`` for select fields in the same config layout."""
    options: Dict[str, Dict[str, str]] = {}
    for item in _iter_config_fields(config):
        if item.get("type") == "select" and item.get("options"):
            options[item["name"]] = {str(value): str(label) for value, label in item["options"]}
    return options


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.6g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value).strip()


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _format_table(rows: Sequence[Dict[str, Any]], max_rows: int = MAX_TABLE_ROWS) -> List[str]:
    """Render a list of homogeneous dicts as compact pipe-delimited text,
    keeping the head and tail when the table is long."""
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return []
    columns = [key for key, value in rows[0].items() if _is_scalar(value)]
    if not columns:
        return []
    # JSON round-trips through the browser sort keys alphabetically; put the
    # identifying columns (dates, periods, labels) back in front.
    leading = [col for col in columns if any(tag in col.lower() for tag in ("date", "period", "label", "scenario"))]
    columns = leading + [col for col in columns if col not in leading]
    lines = [" | ".join(label_for(col) for col in columns)]
    if len(rows) <= max_rows:
        selected: List[Optional[Dict[str, Any]]] = list(rows)
    else:
        head = max_rows * 2 // 3
        tail = max_rows - head
        selected = list(rows[:head]) + [None] + list(rows[-tail:])
    for row in selected:
        if row is None:
            lines.append(f"... ({len(rows) - max_rows} rows omitted) ...")
            continue
        lines.append(" | ".join(format_value(row.get(col)) for col in columns))
    return lines


def _format_delimited_schedule(raw: str, headers: Sequence[str], max_rows: int = MAX_SCHEDULE_ROWS) -> List[str]:
    """Format the ``a|b|c;a|b|c`` schedule strings stored in hidden form fields."""
    rows = [row.strip() for row in str(raw or "").replace("\n", ";").split(";") if row.strip()]
    if not rows:
        return []
    lines = [" | ".join(headers)]
    if len(rows) > max_rows:
        head = max_rows * 2 // 3
        tail = max_rows - head
        lines.extend(rows[:head])
        lines.append(f"... ({len(rows) - max_rows} rows omitted) ...")
        lines.extend(rows[-tail:])
    else:
        lines.extend(rows)
    return lines


def default_input_formatter(spec: AssessmentProductSpec) -> InputFormatter:
    def _format(inputs: Dict[str, Any]) -> List[Tuple[str, str]]:
        inputs = inputs or {}
        # Known fields first, in the order the form defines them; anything else after.
        ordered_keys = [key for key in spec.input_labels if key in inputs]
        ordered_keys += [key for key in inputs if key not in spec.input_labels]

        pairs: List[Tuple[str, str]] = []
        for key in ordered_keys:
            value = inputs[key]
            if key in spec.skip_inputs or value is None or value == "":
                continue
            if isinstance(value, (list, tuple)):
                value = ", ".join(format_value(item) for item in value)
            elif isinstance(value, dict):
                value = "; ".join(f"{k}={format_value(v)}" for k, v in value.items())
            text = format_value(value)
            option_label = spec.input_options.get(key, {}).get(str(value))
            if option_label and option_label != text:
                text = f"{option_label} ({text})"
            pairs.append((label_for(key, spec.input_labels), text))
        return pairs

    return _format


def default_output_formatter(spec: AssessmentProductSpec) -> OutputFormatter:
    def _format(outputs: Dict[str, Any]) -> List[AssessmentSection]:
        sections: List[AssessmentSection] = []
        outputs = outputs or {}

        primary = outputs.get("primary_metrics")
        if isinstance(primary, list):
            section = AssessmentSection("Primary metrics (as displayed):")
            for metric in primary:
                if isinstance(metric, dict) and "label" in metric:
                    section.lines.append(f"- {metric['label']}: {format_value(metric.get('value'))}")
            sections.append(section)

        benchmark = outputs.get("benchmark_metrics")
        if isinstance(benchmark, list):
            section = AssessmentSection("Unrounded metrics:")
            for metric in benchmark:
                if isinstance(metric, dict) and "label" in metric:
                    section.lines.append(f"- {metric['label']}: {format_value(metric.get('value'))}")
            sections.append(section)

        scenarios = outputs.get("scenarios")
        if isinstance(scenarios, list) and scenarios:
            section = AssessmentSection("Scenario results:")
            for scenario in scenarios:
                if not isinstance(scenario, dict):
                    continue
                label = scenario.get("label", "Scenario")
                scalars = [
                    f"{label_for(k)}={format_value(v)}"
                    for k, v in scenario.items()
                    if k != "label" and _is_scalar(v)
                ]
                section.lines.append(f"- {label}: " + ", ".join(scalars))
            sections.append(section)

        if outputs.get("price_decomposition"):
            section = AssessmentSection("Price decomposition:")
            for k, v in outputs["price_decomposition"].items():
                if _is_scalar(v):
                    section.lines.append(f"- {label_for(k)}: {format_value(v)}")
            sections.append(section)

        handled = {"primary_metrics", "benchmark_metrics", "scenarios", "price_decomposition", "summary"}
        for key, value in outputs.items():
            if key in handled or key in spec.skip_outputs:
                continue
            if isinstance(value, list) and value and isinstance(value[0], dict):
                table = _format_table(value)
                if table:
                    sections.append(AssessmentSection(f"{label_for(key)} ({len(value)} rows):", table))
            elif isinstance(value, dict):
                scalars = [f"- {label_for(k)}: {format_value(v)}" for k, v in value.items() if _is_scalar(v)]
                if scalars:
                    sections.append(AssessmentSection(f"{label_for(key)}:", scalars))
            elif _is_scalar(value):
                sections.append(AssessmentSection(f"{label_for(key)}:", [format_value(value)]))

        if outputs.get("summary"):
            sections.append(AssessmentSection("Model summary note:", [str(outputs["summary"])]))
        return sections

    return _format


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_assessment_prompt(
    spec: AssessmentProductSpec,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
) -> str:
    """Compose the full prompt: role, product, inputs, outputs, instructions."""
    input_formatter = spec.input_formatter or default_input_formatter(spec)
    output_formatter = spec.output_formatter or default_output_formatter(spec)

    input_lines = [f"- {label}: {value}" for label, value in input_formatter(inputs)]
    output_sections = [section.render() for section in output_formatter(outputs)]
    output_sections = [block for block in output_sections if block]

    parts = [
        "You are a senior quantitative analyst and model risk reviewer assessing valuation "
        "output from DerivaPro, a derivatives and fixed-income pricing application.",
        f"Product: {spec.title}",
        "",
        "INPUT PARAMETERS (what the user submitted):",
        "\n".join(input_lines) if input_lines else "(no inputs supplied)",
        "",
        "MODEL OUTPUTS (what the pricing engine produced):",
        "\n\n".join(output_sections) if output_sections else "(no outputs supplied)",
        "",
        "TASK:",
        "Using the input parameters as context, assess the model outputs. Specifically:",
        "1. State whether the outputs are consistent with and reasonable given the inputs "
        "(for example, price versus coupon/yield relationship, duration and convexity versus "
        "maturity and amortization profile, scenario symmetry and sign).",
        "2. Identify the main drivers of the result and quantify them where the numbers allow.",
        "3. Flag anything that looks anomalous, internally inconsistent, or that a validator "
        "should follow up on.",
        "4. Note key limitations or caveats of the methodology implied by the inputs.",
    ]
    if spec.focus:
        parts.append(f"Product-specific focus: {spec.focus}")
    parts.extend(
        [
            "",
            f"Cite the actual figures. Keep the response under {spec.word_limit} words, in plain prose "
            "with short paragraphs or brief bullet points. Do not use markdown headings and do not "
            "restate the inputs verbatim.",
        ]
    )
    prompt = "\n".join(parts)
    if len(prompt) > MAX_PROMPT_CHARS:
        logger.warning(
            "AI assessment prompt for %s truncated from %d to %d characters",
            spec.key,
            len(prompt),
            MAX_PROMPT_CHARS,
        )
        prompt = prompt[:MAX_PROMPT_CHARS] + "\n...(context truncated)..."
    return prompt


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def configured_model_name() -> Optional[str]:
    from ..llm.atlas_auth import env_value

    return env_value("LLM_MODEL", "Model")


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    if "403" in message:
        return "Access to the AI service is currently restricted. Please verify the API configuration or contact support."
    if "401" in message:
        return "Authentication with the AI service failed. Please verify the API credentials."
    if "429" in message:
        return "The AI service is rate limited right now. Please wait a moment and try again."
    if "Unsupported LLM provider" in message or "required for" in message:
        return f"The AI service is not configured correctly: {message}"
    return f"The AI service returned an error: {message}"


def generate_assessment(
    spec: AssessmentProductSpec,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    model: Optional[str] = None,
) -> AssessmentResult:
    """Build the prompt and call the configured provider.

    Never raises: configuration and transport errors are folded into
    ``AssessmentResult.error`` so route handlers can return a clean JSON body.
    """
    prompt = build_assessment_prompt(spec, inputs, outputs)
    model_name = model or configured_model_name()
    provider_name = None
    try:
        info = llm_client.get_model_info()
        provider_name = info.get("provider") if isinstance(info, dict) else None
        text = llm_client.generate_response(prompt=prompt, model=model_name)
        text = (text or "").strip()
        if not text:
            raise RuntimeError("The AI service returned an empty response.")
        return AssessmentResult(
            ok=True,
            assessment=text,
            model=model_name,
            provider=provider_name,
            prompt_chars=len(prompt),
        )
    except Exception as exc:  # noqa: BLE001 - surface every failure to the UI
        logger.exception("AI assessment failed for product %s", spec.key)
        return AssessmentResult(
            ok=False,
            assessment="",
            model=model_name,
            provider=provider_name,
            prompt_chars=len(prompt),
            error=_friendly_error(exc),
        )


# ---------------------------------------------------------------------------
# Persistence helpers (optional; used by the JSON endpoint)
# ---------------------------------------------------------------------------


def assessment_payload(result: AssessmentResult, spec: AssessmentProductSpec) -> Dict[str, Any]:
    """The JSON stored in ``AnalysisResult.result_json`` for a saved assessment."""
    return {
        "product_key": spec.key,
        "product_title": spec.title,
        "assessment": result.assessment,
        "model": result.model,
        "provider": result.provider,
        "prompt_chars": result.prompt_chars,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def schedule_input_formatter(
    spec: AssessmentProductSpec,
    schedule_fields: Dict[str, Sequence[str]],
) -> InputFormatter:
    """Input formatter that expands ``a|b;a|b`` hidden schedule fields into
    readable tables and otherwise behaves like the default formatter.

    ``schedule_fields`` maps the form key to its column headers.
    """
    base = default_input_formatter(
        AssessmentProductSpec(
            key=spec.key,
            title=spec.title,
            input_labels=spec.input_labels,
            input_options=spec.input_options,
            skip_inputs=frozenset(spec.skip_inputs) | frozenset(schedule_fields),
        )
    )

    def _format(inputs: Dict[str, Any]) -> List[Tuple[str, str]]:
        pairs = base(inputs)
        for key, headers in schedule_fields.items():
            lines = _format_delimited_schedule(inputs.get(key, ""), headers)
            if lines:
                pairs.append((label_for(key, spec.input_labels), "\n    " + "\n    ".join(lines)))
        return pairs

    return _format


__all__ = [
    "ANALYSIS_TYPE",
    "AssessmentProductSpec",
    "AssessmentResult",
    "AssessmentSection",
    "assessment_payload",
    "build_assessment_prompt",
    "configured_model_name",
    "default_input_formatter",
    "default_output_formatter",
    "generate_assessment",
    "get_assessment_product",
    "labels_from_field_sections",
    "options_from_field_sections",
    "register_assessment_product",
    "registered_assessment_products",
    "schedule_input_formatter",
]
