from causal_model.graph import build_dag
from causal_model.identification import identify_with_dowhy


def test_ate_identified_by_backdoor(df, contract):
    result = identify_with_dowhy(df, contract, build_dag(contract))
    assert result["identified"]
    assert result["strategy"] == "backdoor"
    assert sorted(result["backdoor_variables"]) == sorted(contract["causal"]["pre_treatment_covariates"])
