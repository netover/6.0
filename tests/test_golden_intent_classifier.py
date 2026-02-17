import json
import pytest
from pathlib import Path
from resync.core.agent_router import IntentClassifier, Intent, RoutingMode

def load_golden_cases():
    cases_path = Path(__file__).parent / "golden" / "intent_classifier_cases.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def classifier():
    return IntentClassifier()

@pytest.mark.parametrize("case", load_golden_cases())
def test_intent_classification_golden_cases(classifier, case):
    """
    Validate IntentClassifier against golden truth cases.
    """
    message = case["input"]
    expected_intent = case["expected_intent"]
    expected_routing = case.get("expected_routing")
    expected_entities = case.get("entities", {})

    result = classifier.classify(message)

    # Validate Primary Intent
    assert result.primary_intent.value == expected_intent, \
        f"Failed intent for input: '{message}'. Expected {expected_intent}, got {result.primary_intent.value}"

    # Validate Routing Mode (if specified)
    if expected_routing:
        assert result.suggested_routing.value == expected_routing, \
            f"Failed routing for input: '{message}'. Expected {expected_routing}, got {result.suggested_routing.value}"

    # Validate Entities (subset check)
    for entity_type, expected_values in expected_entities.items():
        assert entity_type in result.entities, f"Missing entity type '{entity_type}' for input: '{message}'"
        
        # Check if all expected values are present in extracted entities
        # Note: regex extraction might be order-independent or have slightly different formats
        extracted = result.entities[entity_type]
        for val in expected_values:
            assert any(val.lower() in str(ext).lower() for ext in extracted), \
                f"Value '{val}' not found in entities for '{message}'. Got {extracted}"

def test_intent_classifier_thresholds(classifier):
    """Test confidence thresholds and classification logic directly."""
    # High confidence greeting
    res = classifier.classify("Oi")
    assert res.is_high_confidence or res.confidence > 0.5
    
    # Low confidence/Unknown
    res = classifier.classify("xpto random sequence")
    assert res.primary_intent == Intent.UNKNOWN
    assert res.needs_clarification
