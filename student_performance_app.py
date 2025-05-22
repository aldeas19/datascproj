
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

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

# Menu lateral para navegação
st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Selecione a página", 
                               ["🏠 Visão Geral", "📊 Análise Exploratória", "🔮 Previsão", "📈 Resultados do Modelo"])

# Carregar modelo e dados
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/random_forest_model.pkl')
        df = pd.read_csv("data/raw/student-data-raw.csv")
        return model, df
    except Exception as e:
        st.error(f"Erro ao carregar artefatos: {str(e)}")
        return None, None

model, df = load_artifacts()

# Páginas da aplicação
if app_mode == "🏠 Visão Geral":
    st.header("Visão Geral do Projeto")
    st.markdown("""
    ### Objetivo Principal
    Desenvolver um sistema de intervenção acadêmica usando machine learning para prever se um aluno 
    do ensino secundário passará ou chumbará no exame final.
    """)

elif app_mode == "📊 Análise Exploratória":
    st.header("Análise Exploratória de Dados (EDA)")

    # Seção 1: Distribuição da Variável Target
    st.subheader("Distribuição de Resultados")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x='passed', data=df, order=['no', 'yes'], ax=ax)
        ax.set_title('Contagem de Resultados')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        df['passed'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], ax=ax)
        ax.set_ylabel('')
        ax.set_title('Proporção Passou/Chumbou')
        st.pyplot(fig)

elif app_mode == "🔮 Previsão":
    st.header("Previsão de Desempenho Estudantil")

    # Formulário para entrada de dados
    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Idade", 15, 22, 17)
            failures = st.selectbox("Reprovações Anteriores", [0, 1, 2, 3])
            absences = st.slider("Faltas", 0, 30, 4)

        with col2:
            studytime = st.selectbox("Tempo de Estudo Semanal", [1, 2, 3, 4], 
                                   format_func=lambda x: ["<2h", "2-5h", "5-10h", ">10h"][x-1])
            Medu = st.selectbox("Educação da Mãe", [0, 1, 2, 3, 4], 
                              format_func=lambda x: ["Nenhuma", "Primário", "5º-9º", "Secundário", "Superior"][x])

        submitted = st.form_submit_button("Prever Desempenho")

    if submitted and model is not None:
        # Criar dataframe com os inputs (simplificado)
        input_data = {
            'age': age,
            'failures': failures,
            'absences': absences,
            'studytime': studytime,
            'Medu': Medu,
            # Adicione outras features conforme necessário
        }

        df_input = pd.DataFrame([input_data])

        # Fazer previsão (ajuste conforme seu pipeline)
        prediction = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1]

        # Exibir resultados
        if prediction == 1:
            st.success(f"✅ Probabilidade de passar: {proba*100:.1f}%")
        else:
            st.error(f"❌ Probabilidade de chumbar: {(1-proba)*100:.1f}%")

elif app_mode == "📈 Resultados do Modelo":
    st.header("Desempenho do Modelo")

    # Matriz de Confusão
    st.subheader("Matriz de Confusão")
    cm = np.array([[30, 10], [5, 35]])  # Substitua por seus valores reais
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Curva ROC
    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])  # Use seus dados reais
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    st.pyplot(fig)
