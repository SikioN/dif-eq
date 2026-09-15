# dif-eq/tools/tests/test_verify.py
from prompts.schema import GeneratedProblem
from verify import check_equilibria

GOODWIN_PROBLEM = {
    "track": "business_informatics",
    "lab_number": 2,
    "title": "Модель Гудвина",
    "system": {
        "variables": ["x", "y"],
        "equations": ["x*(alpha - beta*y)", "-y*(gamma - delta*x)"],
        "parameters": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0},
    },
    "expected_equilibria": [
        {"point": [0.0, 0.0], "type": "saddle"},
        {"point": [1.0, 1.0], "type": "center"},
    ],
    "narrative": "test fixture",
}


def test_correct_classification_accepted():
    problem = GeneratedProblem.model_validate(GOODWIN_PROBLEM)
    result = check_equilibria(problem)
    assert result.accepted, result.reasons


def test_wrong_claimed_type_rejected():
    bad = {
        **GOODWIN_PROBLEM,
        "expected_equilibria": [
            {"point": [0.0, 0.0], "type": "saddle"},
            {"point": [1.0, 1.0], "type": "stable_focus"},  # actually a center
        ],
    }
    problem = GeneratedProblem.model_validate(bad)
    result = check_equilibria(problem)
    assert not result.accepted
    assert any("stable_focus" in r for r in result.reasons)


def test_wrong_equilibrium_count_rejected():
    bad = {
        **GOODWIN_PROBLEM,
        "expected_equilibria": [{"point": [0.0, 0.0], "type": "saddle"}],
    }
    problem = GeneratedProblem.model_validate(bad)
    result = check_equilibria(problem)
    assert not result.accepted
    assert any("expected 1" in r for r in result.reasons)
