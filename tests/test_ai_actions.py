import re

from derivapro.services.ai_assessment import (
    AI_TASKS,
    AssessmentProductSpec,
    build_assessment_prompt,
)


def _spec():
    return AssessmentProductSpec(
        key="callable-amortizing-bond",
        title="Callable Amortizing Bond",
        product_type="fixed_income_callable-amortizing-bond",
        input_labels={
            "coupon_rate": "Coupon Rate",
            "short_rate_volatility_pct": "Short-Rate Volatility (%)",
        },
        methodology_doc="callable_amortizing_bond",
        focus="Check the embedded option value and exercise behavior.",
    )


def _context():
    inputs = {"coupon_rate": "0.055", "short_rate_volatility_pct": "10.00"}
    outputs = {
        "primary_metrics": [
            {"label": "Option-Adjusted Clean PV", "value": "$86.50"},
            {"label": "Embedded Option Value", "value": "$15.25"},
        ]
    }
    return inputs, outputs


def test_task_catalog_exposes_the_four_product_actions():
    assert [item["label"] for item in AI_TASKS.values()] == [
        "Explain results",
        "Create a scenario",
        "Draft report commentary",
        "Ask about methodology",
    ]


def test_each_action_builds_task_specific_prompt():
    inputs, outputs = _context()

    explain = build_assessment_prompt(_spec(), inputs, outputs)
    scenario = build_assessment_prompt(
        _spec(),
        inputs,
        outputs,
        task="create_scenario",
        user_instruction="Use a parallel 100 bp upward curve shock.",
    )
    report = build_assessment_prompt(
        _spec(), inputs, outputs, task="draft_report_commentary"
    )

    assert "Selected action: Explain results" in explain
    assert "Option-Adjusted Clean PV: $86.50" in explain
    assert "Selected action: Create a scenario" in scenario
    assert "current value -> proposed value" in scenario
    assert "parallel 100 bp upward curve shock" in scenario
    assert "must be applied and repriced" in scenario
    assert "Selected action: Draft report commentary" in report
    assert "Do not claim the model is validated" in report


def test_methodology_action_is_grounded_in_repository_document():
    inputs, outputs = _context()
    prompt = build_assessment_prompt(
        _spec(),
        inputs,
        outputs,
        task="ask_methodology",
        user_instruction="How is Bermudan exercise represented?",
    )

    assert "Selected action: Ask about methodology" in prompt
    assert "REPOSITORY METHODOLOGY SOURCE" in prompt
    assert "Callable and Putable Amortizing Bonds" in prompt
    assert "How is Bermudan exercise represented?" in prompt
    assert "If the source does not answer the question" in prompt


def _csrf_token(client):
    response = client.get("/")
    match = re.search(
        r'<meta name="csrf-token" content="([^"]+)">',
        response.get_data(as_text=True),
    )
    assert match is not None
    return match.group(1)


def test_ai_endpoint_validates_task_requirements(authenticated_client):
    endpoint = "/api/ai-assessment/callable-amortizing-bond"
    headers = {"X-CSRFToken": _csrf_token(authenticated_client)}

    unsupported = authenticated_client.post(
        endpoint,
        json={"task": "trade_for_me", "inputs": {}, "outputs": {}},
        headers=headers,
    )
    assert unsupported.status_code == 400
    assert "Unsupported AI action" in unsupported.get_json()["error"]

    missing_outputs = authenticated_client.post(
        endpoint,
        json={"task": "explain_results", "inputs": {}, "outputs": {}},
        headers=headers,
    )
    assert missing_outputs.status_code == 400
    assert "requires pricing outputs" in missing_outputs.get_json()["error"]

    missing_question = authenticated_client.post(
        endpoint,
        json={"task": "ask_methodology", "inputs": {}, "outputs": {}},
        headers=headers,
    )
    assert missing_question.status_code == 400
    assert "methodology question" in missing_question.get_json()["error"]
