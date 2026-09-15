import pytest
from pydantic import ValidationError

from prompts.schema import GeneratedProblem

VALID_PROBLEM = {
    "track": "business_informatics",
    "lab_number": 2,
    "title": "Модель Гудвина: рыночная конкуренция",
    "system": {
        "variables": ["x", "y"],
        "equations": ["x*(alpha - beta*y)", "-y*(gamma - delta*x)"],
        "parameters": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0},
    },
    "expected_equilibria": [
        {"point": [0, 0], "type": "saddle"},
        {"point": [1, 1], "type": "center"},
    ],
    "narrative": "Динамика цены и объёма выпуска на конкурентном рынке.",
}


def test_valid_problem_parses():
    problem = GeneratedProblem.model_validate(VALID_PROBLEM)
    assert problem.track == "business_informatics"
    assert len(problem.expected_equilibria) == 2


def test_missing_field_rejected():
    broken = {k: v for k, v in VALID_PROBLEM.items() if k != "system"}
    with pytest.raises(ValidationError):
        GeneratedProblem.model_validate(broken)


def test_unknown_track_rejected():
    bad = {**VALID_PROBLEM, "track": "underwater_basket_weaving"}
    with pytest.raises(ValidationError):
        GeneratedProblem.model_validate(bad)
