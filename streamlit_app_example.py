
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🎓 Previsão de Aprovação do Aluno")

# Load model
model_data = joblib.load('model_streamlit.pkl')
pipeline = model_data['pipeline']
expected_features = model_data['feature_names']

# Input widgets
st.header("📝 Insira os dados do aluno")
G1 = st.slider("Nota G1", 0, 20, 10)
G2 = st.slider("Nota G2", 0, 20, 10)
absences = st.number_input("Faltas", 0, 100, 4)
failures = st.selectbox("Reprovações anteriores", [0, 1, 2, 3])

# Create FULL input DataFrame with ALL expected features
input_data = {feature: 0 for feature in expected_features}  # Initialize all to 0

# Update with user-provided values
input_data.update({
    'G1': G1,
    'G2': G2,
    'absences': absences,
    'failures': failures,
    # Add reasonable defaults for other important features:
    'Medu': 2,  # Medium education mother (0-4 scale)
    'Fedu': 2,  # Medium education father
    'health': 3,  # Average health
    'famrel': 4,  # Good family relations
})

input_df = pd.DataFrame([input_data], columns=expected_features)

# Prediction
try:
    prediction = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    st.success(f"📊 Probabilidade de passar: {prob:.2%}")
    st.success("✅ O aluno provavelmente passará!" if prediction else "❌ O aluno pode reprovar.")

except Exception as e:
    st.error(f"Erro na predição: {str(e)}")
    st.write("Features esperadas:", expected_features)
    st.write("Features recebidas:", input_df.columns.tolist())
