
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

st.title("🎓 Previsão de Aprovação do Aluno")

# Load model and feature names
try:
    model_data = joblib.load('model_streamlit.pkl')
    pipeline = model_data['pipeline']
    expected_features = model_data['feature_names']

    # Debug info
    st.sidebar.write("⚙️ Model Info:")
    st.sidebar.write(f"Model type: {type(pipeline.named_steps['clf'])}")
    st.sidebar.write(f"Expected features: {expected_features}")

except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Input widgets
st.header("📝 Insira os dados do aluno")
col1, col2 = st.columns(2)
with col1:
    G1 = st.slider("Nota G1", 0, 20, 10)
    absences = st.number_input("Faltas", 0, 100, 4)
with col2:
    G2 = st.slider("Nota G2", 0, 20, 10)
    failures = st.selectbox("Reprovações anteriores", [0, 1, 2, 3])

# Create input with EXACT feature order
try:
    input_df = pd.DataFrame([[0]*len(expected_features)], columns=expected_features)

    # Map inputs to correct columns (adjust based on your actual feature names)
    input_df['G1'] = G1
    input_df['G2'] = G2
    input_df['absences'] = absences
    input_df['failures'] = failures

    # Set default values for other features
    for col in input_df.columns:
        if col not in ['G1', 'G2', 'absences', 'failures']:
            input_df[col] = 0  # Or appropriate default value

except Exception as e:
    st.error(f"Error creating input DataFrame: {str(e)}")
    st.stop()

# Prediction
try:
    prediction = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    # Results display
    st.header("📊 Resultados")
    st.metric("Probabilidade de passar", f"{prob:.2%}")

    if prediction:
        st.success("✅ O aluno provavelmente passará!")
    else:
        st.error("❌ O aluno pode reprovar.")

    # Show feature importance if available
    if hasattr(pipeline.named_steps['clf'], 'feature_importances_'):
        st.header("📌 Importância das Características")
        importances = pipeline.named_steps['clf'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': expected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        st.bar_chart(importance_df.set_index('Feature').head(10))

except Exception as e:
    st.error(f"Error making prediction: {str(e)}")
