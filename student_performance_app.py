import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sistema de Previsão Estudantil", page_icon="🎓")

st.title("🎓 Sistema de Previsão de Desempenho Estudantil")

# Função para carregar pipeline salvo
@st.cache_resource
def load_pipeline():
    return joblib.load("models/student_success_pipeline.pkl")

pipeline = load_pipeline()

st.sidebar.header("Informe os dados do aluno")

# Aqui você coloca os inputs que seu modelo espera
age = st.sidebar.slider("Idade", 15, 22, 17)
sex = st.sidebar.selectbox("Sexo", ["F", "M"])
address = st.sidebar.selectbox("Endereço", ["U", "R"])
famsize = st.sidebar.selectbox("Tamanho da Família", ["LE3", "GT3"])
Pstatus = st.sidebar.selectbox("Status dos Pais", ["T", "A"])
Medu = st.sidebar.selectbox("Educação da Mãe", [0,1,2,3,4],
                            format_func=lambda x: ["Nenhuma","Primário","5º-9º","Secundário","Superior"][x])
Fedu = st.sidebar.selectbox("Educação do Pai", [0,1,2,3,4],
                            format_func=lambda x: ["Nenhuma","Primário","5º-9º","Secundário","Superior"][x])
failures = st.sidebar.selectbox("Reprovações Anteriores", [0,1,2,3])
absences = st.sidebar.slider("Faltas", 0, 30, 4)
studytime = st.sidebar.selectbox("Tempo de Estudo Semanal", [1,2,3,4],
                                format_func=lambda x: ["<2h", "2-5h", "5-10h", ">10h"][x-1])
goout = st.sidebar.selectbox("Frequência de Saídas", [1,2,3,4,5],
                            format_func=lambda x: ["Muito baixa","Baixa","Média","Alta","Muito alta"][x-1])
internet = st.sidebar.selectbox("Acesso à Internet", ["yes", "no"])
Mjob = st.sidebar.selectbox("Profissão da Mãe", ["teacher","health","services","at_home","other"])
Fjob = st.sidebar.selectbox("Profissão do Pai", ["teacher","health","services","at_home","other"])
reason = st.sidebar.selectbox("Motivo da Escolha da Escola", ["home","reputation","course","other"])
guardian = st.sidebar.selectbox("Responsável pelo Aluno", ["mother","father","other"])
schoolsup = st.sidebar.selectbox("Apoio Escolar Extra", ["yes","no"])
famsup = st.sidebar.selectbox("Apoio Familiar Extra", ["yes","no"])
activities = st.sidebar.selectbox("Atividades Extracurriculares", ["yes","no"])
nursery = st.sidebar.selectbox("Frequentou Creche", ["yes","no"])
romantic = st.sidebar.selectbox("Relacionamento Amoroso", ["yes","no"])

# Montar DataFrame para previsão
input_dict = {
    'age': age,
    'sex': sex,
    'address': address,
    'famsize': famsize,
    'Pstatus': Pstatus,
    'Medu': Medu,
    'Fedu': Fedu,
    'failures': failures,
    'absences': absences,
    'studytime': studytime,
    'goout': goout,
    'internet': internet,
    'Mjob': Mjob,
    'Fjob': Fjob,
    'reason': reason,
    'guardian': guardian,
    'schoolsup': schoolsup,
    'famsup': famsup,
    'activities': activities,
    'nursery': nursery,
    'romantic': romantic
}

df_input = pd.DataFrame([input_dict])

if st.button("Prever desempenho"):
    proba = pipeline.predict_proba(df_input)[0][1]
    st.subheader("Resultado da Previsão")
    
    if proba > 0.7:
        st.success(f"✅ Probabilidade de passar: {proba*100:.1f}%")
    elif proba < 0.4:
        st.error(f"❌ Probabilidade de chumbar: {(1-proba)*100:.1f}%")
    else:
        st.warning(f"⚠️ Probabilidade intermediária: {proba*100:.1f}%")

    # Validações simples, exemplo:
    warnings = []
    if failures > 0 and absences > 10:
        warnings.append("⚠️ Histórico de reprovações e muitas faltas.")
    if Medu == 0 and Fedu == 0:
        warnings.append("⚠️ Pais sem educação formal.")

    for w in warnings:
        st.warning(w)