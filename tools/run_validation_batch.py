import json
from collections import Counter
from pathlib import Path

from generate import GenerationConfig, generate_problem
from verify import verify

TRACKS = [
    "mechatronics",
    "infosec",
    "applied_math",
    "software_eng_neuro",
    "applied_informatics_mobile",
    "business_informatics",
]

# The invariant math core per lab, per tz.md §5 / syl.tex §Приложение А —
# every template's {math_core} placeholder is filled from here, so the
# same core is fed to every track for a given lab number.
MATH_CORE_BY_LAB = {
    1: "методы Эйлера и Рунге-Кутты 2-го порядка для задачи Коши x'(t) = f(t, x(t))",
    2: "матрица Якоби, линеаризация в окрестности точек покоя, классификация особых точек",
    3: "функции Ляпунова, чувствительность к начальным условиям, показатели Ляпунова",
    4: "уравнение Эйлера-Лагранжа, дискретизация функционалов, численная оптимизация",
}


def run_batch(
    config: GenerationConfig,
    prompts_dir: Path,
    out_dir: Path,
    per_track: int = 20,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    reasons: Counter = Counter()
    accepted_problems = []

    for track in TRACKS:
        template_path = Path(prompts_dir) / f"{track}.txt"
        for i in range(per_track):
            stats["generated"] += 1
            lab_number = (i % 4) + 1
            try:
                raw = generate_problem(
                    config,
                    str(template_path),
                    track=track,
                    lab_number=lab_number,
                    math_core=MATH_CORE_BY_LAB[lab_number],
                    seed=i,
                )
            except (json.JSONDecodeError, KeyError) as exc:
                stats["rejected_auto"] += 1
                reasons[f"generation/parse error: {exc}"] += 1
                continue

            result = verify(raw)
            if result.accepted:
                stats["accepted"] += 1
                accepted_problems.append(raw)
            else:
                stats["rejected_auto"] += 1
                for reason in result.reasons:
                    reasons[reason] += 1

    (out_dir / "validation_stats.json").write_text(
        json.dumps({"totals": dict(stats), "reasons": dict(reasons)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "accepted_problems.json").write_text(
        json.dumps(accepted_problems, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dict(stats)


if __name__ == "__main__":
    import os

    cfg = GenerationConfig(
        api_key=os.environ["YC_API_KEY"],
        folder_id=os.environ["YC_FOLDER_ID"],
    )
    result = run_batch(cfg, Path("prompts"), Path("validation_out"), per_track=20)
    print(result)
