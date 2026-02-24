import json
from pathlib import Path

import pytest

from resync.core.agent_router import Intent, IntentClassifier


def load_golden_cases() -> list[dict[str, object]]:
    cases_path = Path(__file__).parent / "golden" / "intent_classifier_cases.json"
    with open(cases_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


@pytest.fixture
def classifier() -> IntentClassifier:
    return IntentClassifier()


@pytest.mark.parametrize("case", load_golden_cases())
def test_intent_classification_golden_cases(
    classifier: IntentClassifier,
    case: dict[str, object],
) -> None:
    """
    Validate IntentClassifier against golden truth cases.
    """
    message = str(case["input"])
    expected_intent = str(case["expected_intent"])
    expected_routing = case.get("expected_routing")
    expected_entities = case.get("entities", {})

    result = classifier.classify(message)

    # Validate Primary Intent
    assert result.primary_intent.value == expected_intent, (
        "Failed intent for input: "
        f"'{message}'. Expected {expected_intent}, got {result.primary_intent.value}"
    )

    # Validate Routing Mode (if specified)
    if expected_routing:
        assert result.suggested_routing.value == str(expected_routing), (
            "Failed routing for input: "
            f"'{message}'. Expected {expected_routing}, "
            f"got {result.suggested_routing.value}"
        )

    # Validate Entities (subset check)
    assert isinstance(expected_entities, dict)
    for entity_type, expected_values in expected_entities.items():
        assert isinstance(entity_type, str)
        assert entity_type in result.entities, (
            f"Missing entity type '{entity_type}' for input: '{message}'"
        )

        # Check if all expected values are present in extracted entities
        # Note: regex extraction might be order-independent or have slightly
        # different formats.
        extracted = result.entities[entity_type]
        assert isinstance(expected_values, list)
        for value in expected_values:
            assert any(str(value).lower() in str(ext).lower() for ext in extracted), (
                "Value '"
                f"{value}' not found in entities for '{message}'. "
                f"Got {extracted}"
            )


def test_intent_classifier_thresholds(classifier: IntentClassifier) -> None:
    """Test confidence thresholds and classification logic directly."""
    # High confidence greeting
    res = classifier.classify("Oi")
    assert res.is_high_confidence or res.confidence > 0.5

    # Low confidence/Unknown
    res = classifier.classify("xpto random sequence")
    assert res.primary_intent == Intent.UNKNOWN
    assert res.needs_clarification
