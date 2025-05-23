from sklearn.pipeline import Pipeline
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sistema de Previsão Estudantil", page_icon="🎓")
st.title("🎓 Sistema de Previsão de Desempenho Estudantil")

@st.cache_resource
def load_pipeline_and_encoders():
    return joblib.load("models/student_success_pipeline.pkl")

full_pipeline, label_encoders = load_pipeline_and_encoders()

# Criar pipeline para predição sem oversampler
predict_pipeline = Pipeline(steps=[
    ('preprocessor', full_pipeline.named_steps['preprocessor']),
    ('classifier', full_pipeline.named_steps['classifier'])
])

# Inputs do usuário (igual ao seu)

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

# Função para aplicar label encoding usando os encoders treinados
def encode_input(input_dict, label_encoders):
    encoded = {}
    for col, val in input_dict.items():
        if col in label_encoders:
            le = label_encoders[col]
            val_str = str(val)
            if val_str in le.classes_:
                encoded[col] = le.transform([val_str])[0]
            else:
                # Valor desconhecido, pode tratar como o valor mais comum ou o primeiro da lista
                encoded[col] = le.transform([le.classes_[0]])[0]
        else:
            encoded[col] = val
    return encoded

# Codificar o input do usuário
encoded_input = encode_input(input_dict, label_encoders)
df_input = pd.DataFrame([encoded_input])

if st.button("Prever desempenho"):
    proba = predict_pipeline.predict_proba(df_input)[0][1]
    st.subheader("Resultado da Previsão")
    
    if proba > 0.7:
        st.success(f"✅ Probabilidade de passar: {proba*100:.1f}%")
    elif proba < 0.4:
        st.error(f"❌ Probabilidade de chumbar: {(1-proba)*100:.1f}%")
    else:
        st.warning(f"⚠️ Probabilidade intermediária: {proba*100:.1f}%")

    warnings = []
    if failures > 0 and absences > 10:
        warnings.append("⚠️ Histórico de reprovações e muitas faltas.")
    if Medu == 0 and Fedu == 0:
        warnings.append("⚠️ Pais sem educação formal.")

    for w in warnings:
        st.warning(w)
