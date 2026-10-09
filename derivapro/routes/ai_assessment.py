"""Generic JSON endpoints behind the reusable AI analysis actions.

Any product page can call these once a spec has been registered with
``derivapro.services.ai_assessment.register_assessment_product``:

``POST /api/ai-assessment/<product_key>``
    Body: ``{"task", "instruction", "inputs", "outputs"}``. Builds a task-specific
    prompt from submitted parameters, model outputs, and repository methodology;
    calls the configured LLM provider
    and returns ``{"ok", "assessment", "model", "provider", "saved"}``.
    For signed-in users the assessment is also stored as an ``AnalysisResult``
    (``analysis_type="ai_assessment"``) against the latest pricing run.

``GET /api/ai-assessment/<product_key>/latest``
    Returns the most recently saved assessment for the current user so the
    page can restore it after a reload.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required

from ..extensions import db
from ..models.db_models import AnalysisResult
from ..services.ai_assessment import (
    ANALYSIS_TYPE,
    DEFAULT_AI_TASK,
    MAX_USER_INSTRUCTION_CHARS,
    assessment_payload,
    generate_assessment,
    get_ai_task,
    get_assessment_product,
)
from .result_state import (
    get_latest_analysis_result_for_user,
    get_latest_pricing_result_for_user,
)

ai_assessment_bp = Blueprint("ai_assessment", __name__)


def _resolve_spec(product_key):
    spec = get_assessment_product(product_key)
    if spec is None:
        return None, (jsonify({"ok": False, "error": f"Unknown product '{product_key}'."}), 404)
    return spec, None


def _persist(spec, result):
    """Store the assessment against the user's latest pricing run. Returns the
    saved row id or ``None`` when there is nothing to attach it to."""
    if not spec.product_type or not current_user.is_authenticated:
        return None
    pricing_result = get_latest_pricing_result_for_user(spec.product_type, current_user.id)
    if pricing_result is None:
        return None
    row = AnalysisResult(
        user_id=current_user.id,
        instrument_id=pricing_result.instrument_id,
        pricing_result_id=pricing_result.id,
        analysis_type=ANALYSIS_TYPE,
        result_json=assessment_payload(result, spec),
    )
    db.session.add(row)
    db.session.commit()
    return row.id


@ai_assessment_bp.route("/api/ai-assessment/<product_key>", methods=["POST"])
@login_required
def generate(product_key):
    spec, error_response = _resolve_spec(product_key)
    if error_response:
        return error_response

    body = request.get_json(silent=True) or {}
    inputs = body.get("inputs") or {}
    outputs = body.get("outputs") or {}
    task = str(body.get("task") or DEFAULT_AI_TASK).strip()
    instruction = body.get("instruction") or ""
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        return jsonify({"ok": False, "error": "'inputs' and 'outputs' must be JSON objects."}), 400
    if not isinstance(instruction, str):
        return jsonify({"ok": False, "error": "'instruction' must be text."}), 400
    task_spec = get_ai_task(task)
    if task_spec is None:
        return jsonify({"ok": False, "error": f"Unsupported AI action '{task}'."}), 400
    instruction = instruction.strip()
    if len(instruction) > MAX_USER_INSTRUCTION_CHARS:
        return jsonify(
            {
                "ok": False,
                "error": f"Task detail must be {MAX_USER_INSTRUCTION_CHARS} characters or fewer.",
            }
        ), 400
    if task_spec["requires_outputs"] and not outputs:
        return jsonify({"ok": False, "error": "Run a valuation first; this action requires pricing outputs."}), 400
    if task_spec["requires_instruction"] and not instruction:
        return jsonify({"ok": False, "error": "Enter a methodology question before running this action."}), 400

    result = generate_assessment(
        spec,
        inputs,
        outputs,
        task=task,
        user_instruction=instruction,
    )
    saved_id = _persist(spec, result) if result.ok else None

    payload = result.to_dict()
    payload.update({"product_key": spec.key, "product_title": spec.title, "saved": saved_id is not None})
    return jsonify(payload), (200 if result.ok else 502)


@ai_assessment_bp.route("/api/ai-assessment/<product_key>/latest", methods=["GET"])
@login_required
def latest(product_key):
    spec, error_response = _resolve_spec(product_key)
    if error_response:
        return error_response
    if not spec.product_type:
        return jsonify({"ok": True, "found": False})

    row = get_latest_analysis_result_for_user(spec.product_type, ANALYSIS_TYPE, current_user.id)
    if row is None or not row.result_json:
        return jsonify({"ok": True, "found": False})

    # Only restore an assessment that belongs to the latest pricing run so the
    # page never shows commentary about stale numbers.
    pricing_result = get_latest_pricing_result_for_user(spec.product_type, current_user.id)
    if pricing_result is not None and row.pricing_result_id not in (None, pricing_result.id):
        return jsonify({"ok": True, "found": False})

    data = dict(row.result_json)
    data.update({"ok": True, "found": True, "saved": True, "saved_at": row.created_at.isoformat() + "Z"})
    return jsonify(data)
