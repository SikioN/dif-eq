# dif-eq/tools/interface_app.py
import sympy as sp
import streamlit as st

from generate import GenerationConfig, generate_problem
from run_validation_batch import MATH_CORE_BY_LAB
from verify import verify

TRACK_LABELS = {
    "🤖 Мехатроника и робототехника": "mechatronics",
    "🔐 Информационная безопасность": "infosec",
    "📐 Прикладная математика и информатика": "applied_math",
    "🧠 Программная инженерия (Нейротехнологии)": "software_eng_neuro",
    "📱 Прикладная информатика (Мобильные технологии)": "applied_informatics_mobile",
    "📈 Бизнес-информатика": "business_informatics",
}

EQUILIBRIUM_TYPE_RU = {
    "stable_node": "устойчивый узел",
    "unstable_node": "неустойчивый узел",
    "saddle": "седло",
    "stable_focus": "устойчивый фокус",
    "unstable_focus": "неустойчивый фокус",
    "center": "центр",
    "degenerate": "вырожденная",
}

st.set_page_config(page_title="МатКонтекст", page_icon="🧮", layout="wide")

st.title("🧮 МатКонтекст")
st.caption(
    "Промпт-шаблон направления → LLM (Yandex AI Studio) → "
    "автоматическая верификация (SymPy/SciPy) → приёмка преподавателем"
)

with st.sidebar:
    st.header("Параметры генерации")
    track_ru = st.selectbox("Направление", list(TRACK_LABELS.keys()))
    lab_number = st.selectbox(
        "Лабораторная работа",
        [1, 2, 3, 4],
        format_func=lambda n: f"№{n}",
    )
    generate_clicked = st.button("✨ Сгенерировать задачу", use_container_width=True)
    st.divider()
    st.caption(
        "Задача не попадёт к студенту без автоматической проверки: "
        "существование точек покоя, классификация по Якобиану, "
        "численная устойчивость схемы."
    )


def render_equation(eq_str: str, names: list[str]) -> str:
    """Best-effort LaTeX rendering of a raw equation string for display only
    (not used for verification — verify.py does that from the raw JSON).
    `names` (variables + parameters) are mapped to fresh Symbols so names
    like `beta` or `gamma` aren't shadowed by SymPy's built-in functions."""
    try:
        local_dict = {name: sp.Symbol(name) for name in names}
        return sp.latex(sp.sympify(eq_str, locals=local_dict))
    except (sp.SympifyError, TypeError, ValueError):
        return eq_str


def render_problem(raw: dict) -> None:
    system = raw.get("system", {})
    variables = system.get("variables", [])

    st.subheader(raw.get("title", "Без названия"))
    badge_col1, badge_col2 = st.columns(2)
    badge_col1.metric("Направление", track_ru.split(" ", 1)[-1])
    badge_col2.metric("Лабораторная работа", f"№{raw.get('lab_number', '?')}")

    if raw.get("narrative"):
        st.markdown(f"> {raw['narrative']}")

    params = system.get("parameters", {})
    names = list(variables) + list(params.keys())

    left, right = st.columns(2)
    with left:
        st.markdown("**Система уравнений**")
        for var, eq in zip(variables, system.get("equations", [])):
            st.latex(rf"\dot{{{var}}} = {render_equation(eq, names)}")

        if params:
            st.markdown("**Параметры**")
            st.table({"значение": params})

    with right:
        equilibria = raw.get("expected_equilibria", [])
        if equilibria:
            st.markdown("**Заявленные точки покоя**")
            st.table(
                {
                    "точка": [str(e.get("point")) for e in equilibria],
                    "тип": [
                        EQUILIBRIUM_TYPE_RU.get(e.get("type"), e.get("type"))
                        for e in equilibria
                    ],
                }
            )

    with st.expander("Полный JSON-ответ модели"):
        st.json(raw)


if generate_clicked:
    track = TRACK_LABELS[track_ru]
    config = GenerationConfig(
        api_key=st.secrets["yc_api_key"],
        folder_id=st.secrets["yc_folder_id"],
    )
    template_path = f"prompts/{track}.txt"
    with st.spinner("Генерация и проверка..."):
        raw = generate_problem(
            config,
            template_path,
            track=track,
            lab_number=lab_number,
            math_core=MATH_CORE_BY_LAB[lab_number],
        )
        result = verify(raw)

    if result.accepted:
        st.success("✅ Задача прошла автоматическую проверку")
        render_problem(raw)
    else:
        st.error("❌ Задача отклонена автоматической проверкой")
        for reason in result.reasons:
            st.write(f"— {reason}")
        with st.expander("Полный JSON-ответ модели (отклонено)"):
            st.json(raw)
else:
    st.info(
        "Выберите направление и лабораторную работу в панели слева, затем "
        "нажмите «Сгенерировать задачу». Инструмент покажет сгенерированную "
        "систему уравнений, параметры и заявленные точки покоя — либо "
        "причину отбраковки, если верификация её не пропустила."
    )
