# dif-eq/tools/verify.py
from dataclasses import dataclass, field

import sympy as sp

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

    found_points = [tuple(sol[v] for v in var_syms) for sol in solutions]

    for expected in problem.expected_equilibria:
        target = tuple(sp.nsimplify(c) for c in expected.point)
        match = next(
            (p for p in found_points if all(sp.simplify(a - b) == 0 for a, b in zip(p, target))),
            None,
        )
        if match is None:
            reasons.append(f"expected equilibrium at {expected.point} not found")
            continue
        actual_type = classify_point(var_syms, eqs, match)
        if actual_type != expected.type:
            reasons.append(
                f"equilibrium {expected.point}: expected type '{expected.type}', "
                f"computed '{actual_type}'"
            )

    return VerificationResult(accepted=not reasons, reasons=reasons)
