# dif-eq/tools/interface_app.py
import sympy as sp
import streamlit as st

from generate import GenerationConfig, generate_problem
from run_validation_batch import MATH_CORE_BY_LAB
from verify import verify

TRACK_LABELS = {
    "Мехатроника и робототехника": "mechatronics",
    "Информационная безопасность": "infosec",
    "Прикладная математика и информатика": "applied_math",
    "Программная инженерия (Нейротехнологии)": "software_eng_neuro",
    "Прикладная информатика (Мобильные технологии)": "applied_informatics_mobile",
    "Бизнес-информатика": "business_informatics",
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

st.set_page_config(page_title="МатКонтекст", page_icon=":material/calculate:", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 880px; padding-top: 3rem; }
    h1 { font-weight: 600; letter-spacing: -0.02em; }
    [data-testid="stSidebar"] { border-right: 1px solid #E3E3E6; }
    blockquote {
        border-left: 2px solid #2F3A4C;
        padding-left: 1rem;
        color: #4A4A4A;
        font-style: normal;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("МатКонтекст")
st.write(
    "Генератор профильных учебных задач для курса «Дифференциальные уравнения». "
    "Каждая постановка проверяется математически, прежде чем попасть к преподавателю."
)

with st.sidebar:
    st.subheader("Параметры")
    track_ru = st.selectbox("Направление", list(TRACK_LABELS.keys()))
    lab_number = st.selectbox(
        "Лабораторная работа",
        [1, 2, 3, 4],
        format_func=lambda n: f"№{n}",
    )
    generate_clicked = st.button(
        "Сгенерировать задачу",
        icon=":material/auto_awesome:",
        use_container_width=True,
    )
    st.divider()
    st.caption(
        "Проверяются: существование точек покоя, тип точки покоя "
        "(узел, седло, фокус, центр), устойчивость численного решения."
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
    st.caption(f"{track_ru} · Лабораторная работа №{raw.get('lab_number', '?')}")

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

    with st.expander("Полные данные"):
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
        st.success("Задача прошла проверку", icon=":material/check_circle:")
        render_problem(raw)
    else:
        st.error("Задача отклонена проверкой", icon=":material/cancel:")
        for reason in result.reasons:
            st.write(reason)
        with st.expander("Полные данные (отклонено)"):
            st.json(raw)
else:
    st.info(
        "Выберите направление и лабораторную работу слева, затем нажмите "
        "«Сгенерировать задачу». Вы увидите систему уравнений, параметры и "
        "заявленные точки покоя — либо причину отбраковки, если задача не "
        "прошла проверку.",
        icon=":material/info:",
    )
