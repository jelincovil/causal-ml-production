import networkx as nx

from causal_model.graph import build_dag, satisfies_backdoor, to_dot, to_gml


def test_dag_structure(contract):
    g = build_dag(contract)
    covs = contract["causal"]["pre_treatment_covariates"]
    assert nx.is_directed_acyclic_graph(g)
    assert set(g.nodes) == {"treatment", "outcome", *covs}
    assert g.has_edge("treatment", "outcome")
    assert all(g.has_edge(x, "treatment") and g.has_edge(x, "outcome") for x in covs)


def test_contract_adjustment_satisfies_backdoor(contract):
    g = build_dag(contract)
    assert satisfies_backdoor(g, "treatment", "outcome", contract["causal"]["pre_treatment_covariates"])


def test_empty_adjustment_fails_backdoor(contract):
    g = build_dag(contract)
    assert not satisfies_backdoor(g, "treatment", "outcome", [])


def test_mediator_in_adjustment_set_is_rejected(contract):
    g = build_dag(contract)
    g.add_edges_from([("treatment", "m"), ("m", "outcome")])
    covs = contract["causal"]["pre_treatment_covariates"]
    assert not satisfies_backdoor(g, "treatment", "outcome", [*covs, "m"])


def test_exports(contract):
    g = build_dag(contract)
    assert "treatment" in to_gml(g)
    assert "X -> T" in to_dot(contract)
