
import streamlit as st
import pandas as pd
import joblib

st.title("🎓 Previsão de Aprovação do Aluno")

# Load model and feature names
model_data = joblib.load('model_streamlit.pkl')
pipeline = model_data['pipeline']
expected_features = model_data['feature_names']

# Input widgets
G1 = st.slider("Nota G1", 0, 20, 10)
G2 = st.slider("Nota G2", 0, 20, 10)
absences = st.number_input("Faltas", 0, 100, 4)
failures = st.selectbox("Reprovações anteriores", [0, 1, 2, 3])

# Create input with EXACT feature order
input_df = pd.DataFrame([[G1, G2, absences, failures]], 
                       columns=expected_features)

# Prediction
prediction = pipeline.predict(input_df)[0]
prob = pipeline.predict_proba(input_df)[0][1]

st.write(f"📊 Probabilidade de passar: {prob:.2%}")
st.success("✅ O aluno provavelmente passará!" if prediction else "❌ O aluno pode reprovar.")
