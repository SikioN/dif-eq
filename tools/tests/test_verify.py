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


# The Goodwin fixture above is a conservative (Hamiltonian) system: even
# perturbing near its "wrong" saddle equilibrium at (0,0) stays bounded
# (closed orbits everywhere in the positive quadrant), so it cannot by
# itself prove _reference_equilibrium is doing useful work. This fixture's
# two equilibria genuinely differ in divergence behavior: perturbing near
# (1,0) (a real saddle -- Jacobian eigenvalues (2,-1)) sends x into a
# finite-time Riccati-type blowup, while perturbing near (-1,0) (a real
# stable node -- eigenvalues (-2,-1)) stays bounded.
DIVERGENT_FIRST_PROBLEM = {
    "track": "business_informatics",
    "lab_number": 2,
    "title": "regression fixture for reference-equilibrium selection",
    "system": {
        "variables": ["x", "y"],
        "equations": ["x**2 - 1", "-y"],
        "parameters": {},
    },
    "expected_equilibria": [
        {"point": [1.0, 0.0], "type": "saddle"},
        {"point": [-1.0, 0.0], "type": "stable_node"},
    ],
    "narrative": "test fixture",
}


def test_verify_picks_stable_reference_over_divergent_first_equilibrium():
    # verify() must use _reference_equilibrium to perturb near (-1,0), the
    # stable_node, not blindly near expected_equilibria[0] = (1,0), the
    # saddle -- otherwise this correct, correctly-classified problem would
    # be wrongly rejected on a spurious numerical-stability failure.
    result = verify(DIVERGENT_FIRST_PROBLEM)
    assert result.accepted, result.reasons


def test_wrong_reference_would_have_rejected_divergent_first_problem():
    # Simulates what verify() would do if _reference_equilibrium did not
    # exist (or were broken) and it blindly used expected_equilibria[0]:
    # forcing the saddle at (1,0) as the reference perturbs to x~1.01,
    # which is firmly in the x>1 region where dx/dt = x**2-1 accelerates
    # without bound (finite-time blowup), so the integrator must reject it.
    problem = GeneratedProblem.model_validate(DIVERGENT_FIRST_PROBLEM)
    result = check_numerical_stability(problem, reference=problem.expected_equilibria[0])
    assert not result.accepted, result.reasons
