
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split

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

# Carregar artefatos
@st.cache_resource
def load_artifacts():
    try:
        # Carregar modelo e pré-processadores
        model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')

        # Carregar dados pré-processados
        df = pd.read_csv('data/processed/student_data_processed.csv')

        # Separar features e target
        X = df.drop(columns=['passed'])
        y = df['passed']

        # Dividir em treino e teste (como no notebook)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return model, scaler, label_encoders, X_train, X_test, y_train, y_test, df

    except Exception as e:
        st.error(f"Erro ao carregar artefatos: {str(e)}")
        return None, None, None, None, None, None, None, None

model, scaler, label_encoders, X_train, X_test, y_train, y_test, df = load_artifacts()

# Menu lateral para navegação
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

elif app_mode == "📊 Análise Exploratória" and df is not None:
    st.header("Análise Exploratória de Dados (EDA)")

    # Converter variáveis categóricas de volta para legibilidade
    df_display = df.copy()
    for col in label_encoders:
        df_display[col] = label_encoders[col].inverse_transform(df[col])

    # Seção 1: Distribuição da Variável Target
    st.subheader("Distribuição de Resultados")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x='passed', data=df_display, ax=ax)
        ax.set_title('Contagem de Resultados')
        ax.set_xticklabels(['Chumbou', 'Passou'])
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        df_display['passed'].value_counts().plot.pie(autopct='%1.1f%%', 
                                                    colors=['#ff9999','#66b3ff'], 
                                                    labels=['Chumbou', 'Passou'], 
                                                    ax=ax)
        ax.set_ylabel('')
        ax.set_title('Proporção Passou/Chumbou')
        st.pyplot(fig)

elif app_mode == "🔮 Previsão" and model is not None:
    st.header("Previsão de Desempenho Estudantil")

    # Formulário para entrada de dados
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
        # Criar dataframe com os inputs (com todas as features necessárias)
        input_data = {
            'school': 0,  # GP codificado como 0
            'sex': 0 if sex == "F" else 1,
            'age': age,
            'address': 0 if address == "U" else 1,
            'famsize': 0 if famsize == "LE3" else 1,
            'Pstatus': 0 if Pstatus == "T" else 1,
            'Medu': Medu,
            'Fedu': Fedu,
            'Mjob': 4,  # 'other' codificado como 4
            'Fjob': 4,  # 'other' codificado como 4
            'reason': 0,  # 'course' codificado como 0
            'guardian': 1,  # 'mother' codificado como 1
            'traveltime': 1,
            'studytime': studytime,
            'failures': failures,
            'schoolsup': 0,  # 'no' codificado como 0
            'famsup': 0,  # 'no' codificado como 0
            'paid': 0,  # 'no' codificado como 0
            'activities': 0,  # 'no' codificado como 0
            'nursery': 0,  # 'no' codificado como 0
            'higher': 1,  # 'yes' codificado como 1
            'internet': 0 if internet == "no" else 1,
            'romantic': 0,  # 'no' codificado como 0
            'famrel': 4,
            'freetime': 3,
            'goout': goout,
            'Dalc': 1,
            'Walc': 2,
            'health': 3,
            'absences': absences
        }

        # Criar DataFrame com a mesma ordem das colunas do modelo
        df_input = pd.DataFrame([input_data])
        df_input = df_input[X_train.columns]  # Garantir a mesma ordem

        # Normalizar os dados
        X_input = scaler.transform(df_input)

        # Fazer previsão
        try:
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1]

            # Exibir resultados
            st.subheader("Resultado da Previsão")

            if prediction == 1:
                st.success(f"✅ Probabilidade de passar: {proba*100:.1f}%")
            else:
                st.error(f"❌ Probabilidade de chumbar: {(1-proba)*100:.1f}%")

            # Barra de probabilidade
            st.progress(int(proba*100))

        except Exception as e:
            st.error(f"Erro ao fazer previsão: {str(e)}")

elif app_mode == "📈 Resultados do Modelo" and model is not None and X_test is not None and y_test is not None:
    st.header("Desempenho do Modelo")

    # Matriz de Confusão
    st.subheader("Matriz de Confusão")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Chumbou', 'Passou'])
    disp.plot(cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Curva ROC
    st.subheader("Curva ROC")
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Curva ROC - Random Forest')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Importância das Features
    if hasattr(model, 'feature_importances_'):
        st.subheader("Importância das Features")

        # Mapear nomes das features
        feature_names = X_train.columns
        feature_importance = pd.Series(model.feature_importances_, index=feature_names)
        top_features = feature_importance.sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_features.values, y=top_features.index, ax=ax)
        ax.set_title("Top 10 Features Mais Importantes")
        st.pyplot(fig)

# Rodapé
st.markdown("---")
st.markdown("""
**Sistema de Previsão de Desempenho Estudantil**  
Desenvolvido como parte do projeto de Machine Learning - © 2023
""")
