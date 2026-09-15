# dif-eq/tools/interface_app.py
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

st.title("МатКонтекст — генератор профильных задач")

track_ru = st.selectbox("Направление", list(TRACK_LABELS.keys()))
lab_number = st.selectbox("Лабораторная работа", [1, 2, 3, 4])

if st.button("Сгенерировать задачу"):
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
        st.success("Задача прошла автоматическую проверку")
        st.json(raw)
    else:
        st.error("Задача отклонена автоматической проверкой")
        for reason in result.reasons:
            st.write(f"— {reason}")
