from typing import Literal
from pydantic import BaseModel, field_validator

Track = Literal[
    "mechatronics",
    "infosec",
    "applied_math",
    "software_eng_neuro",
    "applied_informatics_mobile",
    "business_informatics",
]

EquilibriumType = Literal[
    "stable_node", "unstable_node", "saddle",
    "stable_focus", "unstable_focus", "center", "degenerate",
]


class SystemSpec(BaseModel):
    variables: list[str]
    equations: list[str]
    parameters: dict[str, float]

    @field_validator("equations")
    @classmethod
    def equations_match_variables(cls, equations, info):
        variables = info.data.get("variables")
        if variables and len(equations) != len(variables):
            raise ValueError(
                f"expected {len(variables)} equations for {len(variables)} variables, "
                f"got {len(equations)}"
            )
        return equations


class EquilibriumSpec(BaseModel):
    point: list[float]
    type: EquilibriumType


class GeneratedProblem(BaseModel):
    track: Track
    lab_number: int
    title: str
    system: SystemSpec
    expected_equilibria: list[EquilibriumSpec]
    narrative: str
