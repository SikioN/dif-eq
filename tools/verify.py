# dif-eq/tools/verify.py
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from pydantic import ValidationError
from scipy.integrate import solve_ivp

from prompts.schema import GeneratedProblem, EquilibriumSpec


@dataclass
class VerificationResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)


def _build_symbols(problem: GeneratedProblem):
    var_syms = sp.symbols(problem.system.variables)
    if not isinstance(var_syms, (list, tuple)):
        var_syms = (var_syms,)
    param_syms = {name: sp.Symbol(name) for name in problem.system.parameters}
    local_dict = {name: sym for name, sym in zip(problem.system.variables, var_syms)}
    local_dict.update(param_syms)
    return var_syms, param_syms, local_dict


def parsed_equations(problem: GeneratedProblem):
    var_syms, param_syms, local_dict = _build_symbols(problem)
    raw_eqs = [sp.sympify(e, locals=local_dict) for e in problem.system.equations]
    subs_map = {param_syms[name]: value for name, value in problem.system.parameters.items()}
    eqs = [eq.subs(subs_map) for eq in raw_eqs]
    return var_syms, eqs


def classify_point(var_syms, eqs, point):
    jacobian = sp.Matrix(eqs).jacobian(var_syms)
    jacobian_at_point = jacobian.subs(dict(zip(var_syms, point)))
    eigenvals = jacobian_at_point.eigenvals()

    evs = []
    for value, multiplicity in eigenvals.items():
        evs.extend([sp.nsimplify(value)] * multiplicity)

    if all(sp.im(v) == 0 for v in evs):
        signs = [sp.sign(sp.re(v)) for v in evs]
        if any(s == 0 for s in signs):
            return "degenerate"
        if all(s < 0 for s in signs):
            return "stable_node"
        if all(s > 0 for s in signs):
            return "unstable_node"
        return "saddle"

    real_part = sp.re(evs[0])
    if real_part < 0:
        return "stable_focus"
    if real_part > 0:
        return "unstable_focus"
    return "center"


def check_equilibria(problem: GeneratedProblem) -> VerificationResult:
    reasons: list[str] = []
    var_syms, eqs = parsed_equations(problem)
    solutions = sp.solve(eqs, var_syms, dict=True)

    if len(solutions) != len(problem.expected_equilibria):
        reasons.append(
            f"expected {len(problem.expected_equilibria)} equilibria, found {len(solutions)}"
        )
        return VerificationResult(accepted=False, reasons=reasons)

    found_points = []
    for sol in solutions:
        if any(v not in sol or sol[v].free_symbols for v in var_syms):
            reasons.append(
                "solver returned a non-concrete (parametric) equilibrium — cannot verify"
            )
            return VerificationResult(accepted=False, reasons=reasons)
        found_points.append(tuple(sol[v] for v in var_syms))

    matched_indices: set[int] = set()
    for expected in problem.expected_equilibria:
        target = tuple(sp.nsimplify(c) for c in expected.point)
        match_index = next(
            (
                i
                for i, p in enumerate(found_points)
                if i not in matched_indices
                and all(sp.simplify(a - b) == 0 for a, b in zip(p, target))
            ),
            None,
        )
        if match_index is None:
            reasons.append(f"expected equilibrium at {expected.point} not found")
            continue
        matched_indices.add(match_index)
        actual_type = classify_point(var_syms, eqs, found_points[match_index])
        if actual_type != expected.type:
            reasons.append(
                f"equilibrium {expected.point}: expected type '{expected.type}', "
                f"computed '{actual_type}'"
            )

    return VerificationResult(accepted=not reasons, reasons=reasons)


_STABLE_TYPES = {"stable_node", "stable_focus", "center"}


def _reference_equilibrium(problem: GeneratedProblem) -> EquilibriumSpec:
    """Pick an equilibrium to perturb around for the boundedness check.

    Perturbing near a saddle/unstable point makes the trajectory diverge
    for physical reasons that have nothing to do with the numerical
    scheme's own stability — so prefer a stable/center equilibrium when
    the problem has one, and only fall back to the first equilibrium
    (which may be unstable) if none is available.
    """
    for equilibrium in problem.expected_equilibria:
        if equilibrium.type in _STABLE_TYPES:
            return equilibrium
    return problem.expected_equilibria[0]


def check_numerical_stability(
    problem: GeneratedProblem,
    step: float = 0.01,
    t_max: float = 50.0,
    bound: float = 1e6,
    reference: EquilibriumSpec | None = None,
) -> VerificationResult:
    var_syms, eqs = parsed_equations(problem)
    rhs = sp.lambdify(var_syms, eqs, modules="numpy")

    def ode(_t, y):
        return np.array(rhs(*y), dtype=float)

    reference = reference or problem.expected_equilibria[0]
    y0 = [c + 0.01 for c in reference.point]
    t_eval = np.arange(0, t_max, step)

    solution = solve_ivp(ode, (0, t_max), y0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9)

    reasons: list[str] = []
    if not solution.success:
        reasons.append(f"integration failed: {solution.message}")
    elif np.any(~np.isfinite(solution.y)):
        reasons.append("solution contains NaN/Inf — scheme diverged")
    elif np.max(np.abs(solution.y)) > bound:
        reasons.append(f"solution exceeded bound {bound} — likely unstable at step {step}")

    return VerificationResult(accepted=not reasons, reasons=reasons)


def verify(raw_json: dict) -> VerificationResult:
    try:
        problem = GeneratedProblem.model_validate(raw_json)
    except ValidationError as exc:
        return VerificationResult(accepted=False, reasons=[f"schema validation failed: {exc}"])

    checks = [
        check_equilibria(problem),
        check_numerical_stability(problem, reference=_reference_equilibrium(problem)),
    ]
    all_reasons = [reason for result in checks for reason in result.reasons]
    return VerificationResult(accepted=all(result.accepted for result in checks), reasons=all_reasons)
