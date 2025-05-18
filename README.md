Divisão de Trabalho Paralelo (Todos os 3 Podem Começar ao Mesmo Tempo)

Pessoa 1: Exploração e pré-processamento de dados - ALICE!
Tarefas (podem começar de imediato):
Carregue o conjunto de dados, inspecione as estatísticas básicas (valores em falta, tipos de recursos, equilíbrio de classes).
Gerar análise univariada (histogramas, gráficos de caixa para características numéricas, gráficos de barras para categóricas).
Identifique outliers óbvios, padrões de dados em falta e possíveis redundâncias de recursos.

Saída: Relatório EDA inicial com visualizações (ainda sem pré-processamento).


Pessoa 2: Modelação de base e experiências simples - BIA
Tarefas (podem começar de imediato):
Utilize o conjunto de dados brutos (ou minimamente pré-processados, por exemplo, trate os valores em falta com uma estratégia simples como a imputação de médias).
Treine modelos de base (por exemplo, regressão logística, árvore de decisão, KNN) com parâmetros padrão.
Avalie utilizando uma divisão simples de treino e teste (sem ajustes ainda) para obter uma referência de desempenho.

Saída: pontuações de precisão/recall de base para comparação posterior.


Pessoa 3: Pesquisa e Preparação Avançada - CAROL
Tarefas (podem começar de imediato):
Pesquise o contexto do conjunto de dados (por exemplo, estudos de desempenho dos alunos, enviesamentos conhecidos).
Planeie abordagens de agrupamento (por exemplo, que recursos utilizar para aprendizagem não supervisionada).
Estude as considerações sobre a IA responsável (por exemplo, métricas de justiça para subgrupos demográficos).
Preparar modelos de código para SHAP/LIME (interpretação de modelos) ou PCA/t-SNE (visualização).

Saída: Documentação sobre possíveis métodos avançados para aplicação posterior.


Pontos de sincronização (colaboração necessária posteriormente)
Após EDA inicial (Pessoa 1) + Modelos de base (Pessoa 2):
Decidam em conjunto as etapas de pré-processamento (por exemplo, como lidar com valores discrepantes, necessidades de dimensionamento).
A Pessoa 2 pode refinar os modelos após a limpeza da Pessoa 1.
Após a modelação (Pessoa 2) + Preparação avançada (Pessoa 3):
A Pessoa 3 aplica a análise de agrupamento/justiça no melhor modelo da Pessoa 2.
A pessoa 1 pode ajudar a interpretar a importância do recurso em comparação com as conclusões da EDA.

Interpretação Final (Todas):
Combine insights: Porque é que certos modelos funcionaram? Existem preconceitos? Que recursos são mais importantes?


# Student Performance Prediction

## Project Description
Predicts whether students will pass their final exams based on demographic, social, and school-related features.

## Team Members
- Alice (EDA)
- Bia (Modeling)
- Carol (Advanced Analysis)

## Data Dictionary
| Feature | Description | Type |
|---------|-------------|------|
| school | Student's school (GP or MS) | categorical |
| sex | Student's gender (F or M) | categorical |
| age | Student's age (15-22) | numeric |
| ... | ... | ... |

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run EDA: `python src/eda.py`
3. Preprocess data: `python src/preprocess.py`