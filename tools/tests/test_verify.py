# dif-eq/tools/tests/test_verify.py
from prompts.schema import GeneratedProblem
from verify import check_equilibria, check_numerical_stability, verify

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


def test_duplicate_expected_point_not_double_matched():
    # Both expected entries point at the same actual equilibrium (0,0). The
    # second entry must not silently reuse the first match -- it should come
    # back as "not found", since there is only one (0,0) equilibrium to match
    # and the real second equilibrium (1,1) then goes unclaimed.
    bad = {
        **GOODWIN_PROBLEM,
        "expected_equilibria": [
            {"point": [0.0, 0.0], "type": "saddle"},
            {"point": [0.0, 0.0], "type": "saddle"},
        ],
    }
    problem = GeneratedProblem.model_validate(bad)
    result = check_equilibria(problem)
    assert not result.accepted
    assert any("not found" in r for r in result.reasons)


def test_parametric_solution_rejected_not_crashed():
    # variables=[x, y] but the system is degenerate: "0" is trivially true for
    # any x, and "-y" pins y=0. sp.solve returns a single solution dict {y: 0}
    # with no concrete value for x (a continuum of equilibria along the
    # x-axis), which must be rejected with a clear reason rather than raising
    # KeyError when building found_points.
    degenerate = {
        "track": "business_informatics",
        "lab_number": 2,
        "title": "Degenerate system",
        "system": {
            "variables": ["x", "y"],
            "equations": ["0", "-y"],
            "parameters": {},
        },
        "expected_equilibria": [{"point": [0.0, 0.0], "type": "saddle"}],
        "narrative": "test fixture",
    }
    problem = GeneratedProblem.model_validate(degenerate)
    result = check_equilibria(problem)
    assert not result.accepted
    assert any("non-concrete" in r or "parametric" in r for r in result.reasons)


STABLE_SYSTEM = {
    "track": "business_informatics",
    "lab_number": 2,
    "title": "стабильная линейная система",
    "system": {
        "variables": ["x", "y"],
        "equations": ["-x", "-y"],
        "parameters": {},
    },
    "expected_equilibria": [{"point": [0.0, 0.0], "type": "stable_node"}],
    "narrative": "test fixture",
}

UNSTABLE_SYSTEM = {
    **STABLE_SYSTEM,
    "system": {"variables": ["x", "y"], "equations": ["x", "y"], "parameters": {}},
    "expected_equilibria": [{"point": [0.0, 0.0], "type": "unstable_node"}],
}


def test_bounded_trajectory_accepted():
    problem = GeneratedProblem.model_validate(STABLE_SYSTEM)
    result = check_numerical_stability(problem)
    assert result.accepted, result.reasons


def test_diverging_trajectory_rejected():
    problem = GeneratedProblem.model_validate(UNSTABLE_SYSTEM)
    result = check_numerical_stability(problem, t_max=30.0, bound=1e6)
    assert not result.accepted
    assert any("exceeded bound" in r for r in result.reasons)


def test_verify_rejects_invalid_schema():
    result = verify({"track": "not_a_real_track"})
    assert not result.accepted
    assert any("schema" in r for r in result.reasons)


def test_verify_accepts_correct_problem():
    result = verify(GOODWIN_PROBLEM)
    assert result.accepted, result.reasons
