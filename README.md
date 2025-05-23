# Pipeline de Ciência de Dados para Previsão de Desempenho Académico

Este projeto foi desenvolvido no âmbito da unidade curricular de Elementos de Inteligência Artificial e Ciência de Dados (EIACD).

## Visão Geral do Projeto

O objetivo principal deste trabalho é construir um pipeline completo de ciência de dados, desde a análise exploratória dos dados (EDA) e pré-processamento, até à aplicação de modelos supervisionados de classificação e avaliação dos seus desempenhos. O projeto foca-se na previsão do desempenho académico de estudantes, identificando se serão aprovados ou reprovados com base em características académicas e comportamentais.

Para tal, foram testados quatro modelos de aprendizagem supervisionada: Random Forest, Regressão Logística, KNN e Árvore de Decisão. A avaliação dos modelos foi feita com validação cruzada (K-Fold com k=5), utilizando métricas como Acurácia, Precisão, Recall, F1-Score e ROC-AUC.

## 📌 Objetivos principais

- Análise exploratória (EDA) para compreender padrões nos dados.
- Pré-processamento (dados faltantes, outliers, codificação de variáveis categóricas).
- Treino e avaliação de modelos de classificação (Regressão Logística, Árvores de Decisão, KNN).
- Comparação de métricas de desempenho (precisão, recall, AUC-ROC).
- Interpretação dos resultados para identificar fatores críticos no desempenho dos alunos.

## 📊 Descrição do Dataset :
Utilizamos o UCI Student Performance Dataset, com dados de duas escolas secundárias portuguesas, sendo 33 features que incluem variáveis:
- Demográficas: idade, género, educação dos pais.
- Académicas: notas anteriores (G1, G2), horas de estudo, reprovações.
- Sociais: atividades extracurriculares, apoio escolar.
- Variável alvo: passou (binária, derivada da nota final G3 ≥ 10).
- Tamanho: 649 alunos, sem valores faltantes.

## 🧰 Bibliotecas e Dependências
Este projeto utiliza várias bibliotecas do ecossistema de ciência de dados em Python. Abaixo encontram-se organizadas por finalidade:

### 🔍 Manipulação e Análise de Dados
- pandas: Manipulação de estruturas de dados (DataFrames)
- numpy: Operações matemáticas e vetoriais
- scipy: Ferramentas estatísticas e de computação científica
### 📊 Visualização de Dados
- matplotlib: Geração de gráficos estáticos
- seaborn: Visualizações estatísticas detalhadas
- missingno: Visualização de valores em falta nos dados
### 🧹 Limpeza e Pré-processamento
- pyyaml: Leitura de ficheiros .yaml para configuração
- scikit-learn: Pré-processamento, modelação e avaliação
- imbalanced-learn: Técnicas para lidar com dados desbalanceados
### 💻 Execução e Desenvolvimento
- jupyter e notebook: Notebooks interativos
- nbformat: Manipulação de ficheiros .ipynb
- streamlit: Aplicações web interativas para visualização de resultados
- joblib: Guardar e carregar modelos treinados

## ⚙️ Instalação das dependências 

Segue os passos abaixo para executar o projeto localmente:
### 1. Clonar o repositório
```
git clone https://github.com/aldeas19/datascproj.git
cd datascproj
```

### 2. Criar um ambiente virtual (opcional, mas recomendando)
 ```
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```
Usando requirements.txt:
```
pip install -r requirements.txt
```
Ou manualmente:
```
pip install pandas numpy matplotlib seaborn scipy pyyaml missingno nbformat jupyter scikit-learn imbalanced-learn notebook streamlit joblib
```
## 📓 Execução do Projeto
### Localmente (Jupyter Notebook)

1. Instalar o Jupyter:
```
pip install jupyterlab
```
2. Abrir o notebook:
```
jupyter lab
```
3. Abrir o ficheiro finalnotebook.ipynb e executar as células.

### Google Colab

Fazer upload do ficheiro finalnotebook.ipynb e do conjunto de dados no ambiente do Colab e executar normalmente.

Aplicação com Streamlit (opcional)
```
streamlit run streamlit_app.py
```
# Team Members
- Alice Azevedo Deas 
- Ana Caroline Soares Silva
- Beatriz Moraes Vieira

# Bibliografia 

1. UC Irvine Machine Learning Repository. Student Performance,
https://archive.ics.uci.edu/dataset/320/student+performance
2. Cortez, P., & Silva, A. M. G. (2008). “Using data mining to predict secondary school student performance”. https://repositorium.sdum.uminho.pt/bitstream/1822/8024/1/student.pdf

## Data Dictionary
| Feature | Description | Type |
|---------|-------------|------|
| school | Student's school (GP or MS) | categorical |
| sex | Student's gender (F or M) | categorical |
| age | Student's age (15-22) | numeric |
| ... | ... | ... |


## 📁 Estrutura do Projeto (TO BE CHANGED !)!

- `data/` – Contém os dados em diferentes estágios:
  - `raw/` – Arquivos originais.
  - `processed/` – Prontos para modelagem.

- `notebooks/` – Jupyter Notebooks usados para EDA e testes iniciais:
  - `01-eda.ipynb` – Análise exploratória interativa.
  - `02-modeling.ipynb` – Experimentação de modelos.

- `scripts/` – Scripts em Python que automatizam partes do projeto:
  - `eda.py` – Executa uma EDA completa e salva os gráficos em `/docs`.
  - `preprocess.py` – Limpeza e transformação dos dados.
  - `utils.py` – Funções auxiliares reutilizáveis.
  - `clean_notebook.py` – Limpa outputs de notebooks, útil para controle de versão e submissões.

- `docs/` – Gráficos salvos automaticamente pelo script `eda.py`.

- `config.yml` – Arquivo central de configuração com parâmetros como colunas numéricas, tamanhos de figura, caminhos de arquivos etc.

- `requirements.txt` – Lista de pacotes Python necessários para rodar o projeto.

---

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run EDA: `python src/eda.py`
3. Preprocess data: `python src/preprocess.py`
