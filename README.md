# 🎬 Análise de Sentimentos em Reviews de Filmes
> Classificação automática de sentimentos em reviews do IMDB usando NLP e Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.26-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🔗 **Ver App ao vivo**](https://feelings-project-nlp-bye7qxcohlcbn5odwzhhd6.streamlit.app) | [📓 Notebook EDA](notebooks/01_carregamento_eda.ipynb) | [🤗 Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 📋 Sobre o Projeto

**Problema de negócio:** Empresas que trabalham com avaliações de usuários (e-commerce, streaming, apps) 
recebem milhares de reviews diariamente e não conseguem lê-las manualmente. A classificação automática 
de sentimentos permite identificar rapidamente produtos problemáticos, monitorar satisfação e 
priorizar atendimentos.

**Solução:** Pipeline completo de NLP que lê reviews em texto, limpa e transforma em features numéricas,
treina um modelo de classificação binária e disponibiliza as previsões via aplicativo web interativo.


---

## 🏆 Resultados

| Métrica             |    Valor   |
|---------------------|------------|
| F1-Macro            | **0.9066** |
| Acurácia            | **90.60%** |
| AUC-ROC             | **0.9675** |
| Modelos testados    |      4     |
| Tempo de inferência |  < 100ms   |

---

## 📊 Dataset

- **Fonte:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle)
- **Tamanho:** 50.000 reviews (~50/50 positivas/negativas)
- **Variável alvo:** `sentiment` (positive / negative)
- **Features:** texto bruto da review

---

## 🔧 Metodologia 

```
Dados Brutos → Limpeza de Texto → Feature Engineering → TF-IDF → Modelo → Previsão
```
### Pipeline de NLP
```
1. **Limpeza:** remoção de HTML, pontuação, stopwords, lematização
2. **Feature Engineering:** n_words, n_exclamation, caps_ratio, avg_word_len
3. **Vetorização:** TF-IDF com bigramas (30.000 features)
4. **Modelagem:** 4 algoritmos comparados (Naive Bayes, Logistic Regression, LinearSVC, Random Forest)
5. **Otimização:** RandomizedSearchCV com 5-fold cross-validation

### Modelos Comparados

| Modelo              | F1 CV    | F1 Teste | AUC      |
|---------------------|----------|----------|----------|
| Regressão Logística | 0.899148 | 0.901965 | 0.966110 |
| LinearSVC           | 0.899016 | 0.899361 | 0.964314 |
| Naive Bayes         | 0.879988 | 0.880988 | 0.950022 |
| Random Forest       | 0.843055 | 0.839698 | 0.926138 |

---
```
## 📁 Estrutura do Repositório
```
projeto-sentimentos-nlp/
├── notebooks/
│   ├── 01_carregamento_eda.ipynb      # Análise exploratória
│   ├── 02_preprocessing.ipynb         # Limpeza e features
│   └── 03_modelagem.ipynb             # Treino, avaliação, SHAP
├── src/
│   └── text_utils.py                  # Funções de NLP reutilizáveis
├── reports/figures/                   # Visualizações geradas
├── models/                            # Modelos serializados
├── app.py                             # Aplicativo Streamlit
├── requirements.txt
└── README.md
```
## 🚀 Como Executar Localmente

### Pré-requisitos
- Python 3.9+
- ~2GB de espaço em disco

### Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/Lyraa-Dev/feelings-project-nlp.git
cd projeto-sentimentos-nlp

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe o dataset (necessário para treinar do zero)
# Coloque o arquivo IMDB Dataset.csv em data/raw/

# 5. Execute os notebooks na ordem (01 → 02 → 03)
jupyter notebook

# 6. Inicie o app
streamlit run app.py
```

---

## 📈 Análise Exploratória (Destaques)

**Distribuição balanceada:** O dataset possui 25.000 reviews positivas e 25.000 negativas, 
não necessitando técnicas de balanceamento.

**Comprimento:** Reviews positivas tendem a ser ligeiramente mais longas (média de X palavras vs Y).

**Bigramas informativos:** Bigramas negativos frequentes incluem "waste time", "not recommend", 
"bad movie". Positivos incluem "highly recommend", "well worth", "must see".

---

## 🤝 Contato

**Ricardo Lyra** — [LinkedIn](https://www.linkedin.com/in/lyraa-dev/) — [Email](mailto:lyra.dev001@gmail.com)
