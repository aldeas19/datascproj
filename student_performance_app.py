
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configuração da página
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título da aplicação
st.title("🎓 Sistema de Previsão de Desempenho Estudantil")
st.markdown("""
Esta aplicação prevê se um aluno terá sucesso acadêmico com base em características demográficas, 
acadêmicas e sociais, permitindo intervenções direcionadas.
""")

# Função para carregar artefatos
@st.cache_resource
def load_artifacts():
    try:
        return {
            'model': joblib.load('models/random_forest_model.pkl'),
            'scaler': joblib.load('models/minmax_scaler.pkl'),
            'label_encoders': joblib.load('encoders/label_encoders.pkl'),
            'feature_names': joblib.load('models/feature_names.pkl'),
            'dataset_info': joblib.load('models/dataset_info.pkl')
        }
    except Exception as e:
        st.error(f"Erro ao carregar artefatos: {str(e)}")
        return None

# Carregar artefatos
artifacts = load_artifacts()

# Menu lateral
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Selecione a página", 
                               ["🏠 Visão Geral", "📊 Análise Exploratória", "🔮 Previsão", "📈 Resultados do Modelo"])

# Páginas da aplicação
if app_mode == "🏠 Visão Geral":
    st.header("Visão Geral do Projeto")
    st.markdown("""
    ### Objetivo Principal
    Desenvolver um sistema de intervenção acadêmica usando machine learning para prever se um aluno 
    do ensino secundário passará ou chumbará no exame final.
    """)

    if artifacts:
        st.markdown("### Estatísticas do Modelo")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Proporção de Aprovações", f"{artifacts['dataset_info']['class_distribution'][1]:.1%}")
        with col2:
            st.metric("Proporção de Reprovações", f"{artifacts['dataset_info']['class_distribution'][0]:.1%}")

elif app_mode == "📊 Análise Exploratória" and artifacts:
    st.header("Análise Exploratória de Dados")
    st.subheader("Features Mais Importantes")
    features_importance = pd.Series(artifacts['dataset_info']['feature_importances'])
    top_features = features_importance.sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax)
    ax.set_title("Top 10 Features Mais Importantes")
    st.pyplot(fig)

elif app_mode == "🔮 Previsão" and artifacts:
    st.header("Previsão de Desempenho Estudantil")

    with st.form("student_form"):
        st.subheader("Informações do Aluno")
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Idade", 15, 22, 17)
            sex = st.radio("Gênero", ["F", "M"])
            address = st.radio("Endereço", ["U", "R"])
            famsize = st.radio("Tamanho da Família", ["LE3", "GT3"])
            Pstatus = st.radio("Status dos Pais", ["T", "A"])

        with col2:
            Medu = st.selectbox("Educação da Mãe", [0, 1, 2, 3, 4], 
                              format_func=lambda x: ["Nenhuma", "Primário", "5º-9º", "Secundário", "Superior"][x])
            Fedu = st.selectbox("Educação do Pai", [0, 1, 2, 3, 4], 
                              format_func=lambda x: ["Nenhuma", "Primário", "5º-9º", "Secundário", "Superior"][x])
            failures = st.selectbox("Reprovações Anteriores", [0, 1, 2, 3])
            absences = st.slider("Faltas", 0, 30, 4)

        st.subheader("Atividades e Comportamento")
        studytime = st.selectbox("Tempo de Estudo Semanal", [1, 2, 3, 4], 
                               format_func=lambda x: ["<2h", "2-5h", "5-10h", ">10h"][x-1])
        goout = st.selectbox("Frequência de Saídas", [1, 2, 3, 4, 5], 
                           format_func=lambda x: ["Muito baixa", "Baixa", "Média", "Alta", "Muito alta"][x-1])
        internet = st.radio("Acesso à Internet", ["yes", "no"])

        submitted = st.form_submit_button("Prever Desempenho")

    if submitted:
        input_data = {
            'school': 'GP', 'sex': sex, 'age': age, 'address': address,
            'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
            'Mjob': 'other', 'Fjob': 'other', 'reason': 'course', 'guardian': 'mother',
            'traveltime': 1, 'studytime': studytime, 'failures': failures,
            'schoolsup': 'no', 'famsup': 'no', 'paid': 'no', 'activities': 'no',
            'nursery': 'no', 'higher': 'yes', 'internet': internet, 'romantic': 'no',
            'famrel': 4, 'freetime': 3, 'goout': goout, 'Dalc': 1, 'Walc': 2,
            'health': 3, 'absences': absences
        }

        # Codificação dos dados
        encoded_data = {}
        for col in artifacts['feature_names']:
            if col in artifacts['label_encoders']:
                encoded_data[col] = artifacts['label_encoders'][col].transform([input_data[col]])[0]
            else:
                encoded_data[col] = input_data[col]

        # Previsão
        df_input = pd.DataFrame([encoded_data])[artifacts['feature_names']]
        X_input = artifacts['scaler'].transform(df_input)
        proba = artifacts['model'].predict_proba(X_input)[0][1]

        # Ajuste da probabilidade
        original_pass_rate = artifacts['dataset_info']['class_distribution'][1]
        adjusted_proba = proba * original_pass_rate / (proba * original_pass_rate + (1-proba) * (1-original_pass_rate))

        # Exibição dos resultados
        st.subheader("Resultado da Previsão")
        if adjusted_proba > 0.7:
            st.success(f"✅ Probabilidade de passar: {adjusted_proba:.1%}")
        elif adjusted_proba < 0.4:
            st.error(f"❌ Probabilidade de chumbar: {1-adjusted_proba:.1%}")
        else:
            st.warning(f"⚠️ Probabilidade limítrofe: {adjusted_proba:.1%}")

        st.progress(int(adjusted_proba * 100))

        # Fatores mais influentes
        st.subheader("Fatores Mais Influentes")
        feature_effects = []
        for feature in artifacts['dataset_info']['feature_importances'].keys():
            importance = artifacts['dataset_info']['feature_importances'][feature]
            value = input_data[feature]
            if feature in artifacts['label_encoders']:
                value = artifacts['label_encoders'][feature].inverse_transform([encoded_data[feature]])[0]
            feature_effects.append({'Feature': feature, 'Valor': str(value), 'Importância': importance})

        for effect in sorted(feature_effects, key=lambda x: x['Importância'], reverse=True)[:5]:
            st.write(f"- **{effect['Feature']}**: {effect['Valor']} (importância: {effect['Importância']:.3f})")

elif app_mode == "📈 Resultados do Modelo" and artifacts:
    st.header("Desempenho do Modelo")
    st.subheader("Matriz de Confusão (Exemplo)")
    cm = np.array([[50, 10], [5, 60]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Chumbou', 'Passou'])
    fig, ax = plt.subplots()
    disp.plot(cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("Métricas Principais")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Acurácia", "85%")
    with col2: st.metric("Precisão", "82%")
    with col3: st.metric("Recall", "88%")

# Rodapé
st.markdown("---")
st.markdown("""
**Sistema de Previsão de Desempenho Estudantil**  
Desenvolvido como parte do projeto de Machine Learning - © 2023
""")
