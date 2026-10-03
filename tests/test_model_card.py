import json

from validation.model_card import build_model_card, validate_model_card


def test_committed_model_card_matches_schema():
    from contracts import ROOT

    card = json.loads((ROOT / "artifacts" / "model" / "model_card.json").read_text(encoding="utf-8"))
    assert validate_model_card(card) == []


def test_card_without_required_field_is_rejected():
    assert any("no_usar_para" in e for e in validate_model_card({"schema_version": "causal-model-card-v1"}))
