
import streamlit as st
import pandas as pd
import joblib

st.title("🎓 Previsão de Aprovação do Aluno")

model = joblib.load('model_streamlit.pkl')

# Coleta de inputs
G1 = st.slider("Nota G1", 0, 20, 10)
G2 = st.slider("Nota G2", 0, 20, 10)
absences = st.number_input("Faltas", 0, 100, 4)
failures = st.selectbox("Reprovações anteriores", [0, 1, 2, 3])

# Criação do DataFrame
input_df = pd.DataFrame({
    'G1': [G1],
    'G2': [G2],
    'absences': [absences],
    'failures': [failures]
})

# Aplicação do modelo
prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

st.write(f"📊 Probabilidade de passar: {prob:.2%}")
st.success("✅ O aluno provavelmente passará!" if prediction else "❌ O aluno pode reprovar.")
