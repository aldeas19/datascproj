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
