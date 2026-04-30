
import streamlit as st
import pandas as pd
import joblib
import json
import re
import sys
import os

sys.path.append(os.path.dirname(__file__))
from src.text_utils import clean_text_basic

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Analisador de Sentimentos",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .positive-result {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        color: #2d3436;
    }
    .negative-result {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        color: #2d3436;
    }
</style>
""", unsafe_allow_html=True)


# CARREGAR MODELO (utilizei cache para não recarregar a cada interação)

@st.cache_resource
def carregar_modelo():
    """Carrega vetorizador e modelo serializados."""
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    modelo = joblib.load('models/sentiment_model.pkl')
    
    with open('models/metricas_finais.json') as f:
        metricas = json.load(f)
    
    return tfidf, modelo, metricas

try:
    tfidf, modelo, metricas = carregar_modelo()
    modelo_carregado = True
except FileNotFoundError:
    modelo_carregado = False

# INTERFACE PRINCIPAL

# Header
st.markdown('<p class="main-header">🎬 Analisador de Sentimentos</p>', unsafe_allow_html=True)
st.markdown("**Analise o sentimento de reviews de filmes usando Machine Learning com NLP**")
st.divider()

# Sidebar com informações do modelo
with st.sidebar:
    st.header("📊 Informações do Modelo")
    
    if modelo_carregado:
        st.success("✅ Modelo carregado com sucesso!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F1-Macro", f"{metricas.get('f1_macro_test', 0):.3f}")
        with col2:
            st.metric("AUC-ROC", f"{metricas.get('auc_roc', 0):.3f}")
        
        st.metric("Acurácia", f"{metricas.get('accuracy_test', 0)*100:.1f}%")
        st.metric("Vocabulário TF-IDF", f"{metricas.get('vocab_size', 0):,} termos")
        
        st.divider()
        st.subheader("🔧 Configurações")
        threshold = st.slider("Threshold de Positividade", 
                              min_value=0.3, max_value=0.7, 
                              value=0.5, step=0.05,
                              help="Probabilidade mínima para classificar como positivo")
    else:
        st.error("❌ Modelo não encontrado!\nExecute o notebook 03_modelagem.ipynb primeiro.")
        threshold = 0.5
    
    st.divider()
    st.markdown("**Desenvolvido por:** Ricardo Lyra")
    st.markdown("**Dataset:** IMDB Movie Reviews (50k)")
    st.markdown("**Algoritmo:** Regressão Logística + TF-IDF")

# Tabs principais
tab1, tab2, tab3 = st.tabs(["🔍 Análise Individual", "📋 Análise em Lote", "📈 Sobre o Projeto"])

# ============================================================
# TAB 1: ANÁLISE INDIVIDUAL
# ============================================================
with tab1:
    st.subheader("Analise uma review")
    
    # Exemplos para testar
    exemplos = {
        "Selecione um exemplo...": "",
        "Review Positiva 1": "This film was absolutely amazing! The performances were outstanding and the cinematography breathtaking. I couldn't take my eyes off the screen for a single moment.",
        "Review Negativa 1": "Terrible movie. Complete waste of time and money. The plot makes no sense and the acting is wooden. Avoid at all costs.",
        "Review Ambígua": "The movie had some interesting moments but ultimately failed to deliver on its promise. Not terrible, not great, just mediocre.",
        "Ironia (Difícil)": "Oh wow, what a MASTERPIECE! The director clearly put so much thought into making the worst film of the decade.",
    }
    
    exemplo_selecionado = st.selectbox("Experimente um exemplo:", list(exemplos.keys()))
    
    texto_usuario = st.text_area(
        "Cole ou escreva uma review aqui:",
        value=exemplos[exemplo_selecionado],
        height=180,
        placeholder="Ex: This movie was absolutely brilliant! The story was captivating..."
    )
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analisar = st.button("🔍 Analisar", type="primary", use_container_width=True)
    with col_btn2:
        limpar = st.button("🗑️ Limpar", use_container_width=False)
    
    if limpar:
        st.rerun()
    
    if analisar and texto_usuario.strip():
        if not modelo_carregado:
            st.error("Modelo não encontrado. Execute o notebook de modelagem primeiro.")
        else:
            with st.spinner("Analisando..."):
                # Pré-processar
                texto_limpo = clean_text_basic(texto_usuario)
                
                # Vetorizar e prever
                vetor = tfidf.transform([texto_limpo])
                proba = modelo.predict_proba(vetor)[0]
                prob_positivo = proba[1]
                prob_negativo = proba[0]
                
                # Determinar resultado
                eh_positivo = prob_positivo >= threshold
                
            # Exibir resultado
            st.divider()
            
            if eh_positivo:
                st.markdown(f"""
                <div class="positive-result">
                    <h2>😊 SENTIMENTO POSITIVO</h2>
                    <p>O modelo identificou um sentimento <strong>positivo</strong> nesta review.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="negative-result">
                    <h2>😞 SENTIMENTO NEGATIVO</h2>
                    <p>O modelo identificou um sentimento <strong>negativo</strong> nesta review.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("")
            
            # Métricas de confiança
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probabilidade Positivo", f"{prob_positivo*100:.1f}%",
                          delta=f"{(prob_positivo-0.5)*100:+.1f}% vs baseline")
            with col2:
                st.metric("Probabilidade Negativo", f"{prob_negativo*100:.1f}%")
            with col3:
                confianca = max(prob_positivo, prob_negativo)
                nivel = "Alta" if confianca > 0.8 else "Média" if confianca > 0.65 else "Baixa"
                st.metric("Confiança", f"{confianca*100:.1f}%", delta=nivel)
            
            # Barra de probabilidade
            st.write("**Distribuição de probabilidade:**")
            prob_df = pd.DataFrame({
                "Positivo 😊": [prob_positivo],
                "Negativo 😞": [prob_negativo]
            })
            st.bar_chart(prob_df, color=["#2ecc71", "#eb3a26"])
            
            # Detalhes técnicos (expansível)
            with st.expander("🔬 Ver detalhes técnicos"):
                st.write("**Texto após pré-processamento:**")
                st.code(texto_limpo[:500] + ("..." if len(texto_limpo) > 500 else ""))
                
                # Top palavras influentes nessa review específica
                feature_names = tfidf.get_feature_names_out()
                review_vetor = vetor.toarray()[0]
                tokens_presentes = [(feature_names[i], review_vetor[i])
                                    for i in review_vetor.nonzero()[0]]
                
                if hasattr(modelo, 'coef_'):
                    coefs = modelo.coef_[0]
                    tokens_com_peso = [(tok, val * coefs[i])
                                       for tok, val in tokens_presentes
                                       for i in [list(feature_names).index(tok)]]
                    tokens_com_peso.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    st.write("**Top 10 tokens mais influentes nesta review:**")
                    for tok, peso in tokens_com_peso[:10]:
                        direcao = "→ POSITIVO" if peso > 0 else "→ NEGATIVO"
                        st.write(f"  `{tok}`: {peso:+.4f} {direcao}")
    
    elif analisar and not texto_usuario.strip():
        st.warning("⚠️ Por favor, insira um texto para analisar.")

# ============================================================
# TAB 2: ANÁLISE EM LOTE
# ============================================================
with tab2:
    st.subheader("Analise múltiplas reviews de uma vez")
    st.write("Cole uma review por linha:")
    
    texto_lote = st.text_area(
        "Reviews (uma por linha):",
        height=200,
        placeholder="Horrible movie, waste of time.\nDecent film, nothing special."
    )
    
    if st.button("🔍 Analisar Lote", type="primary"):
        if not modelo_carregado:
            st.error("Modelo não encontrado.")
        elif texto_lote.strip():
            reviews = [r.strip() for r in texto_lote.strip().split('\n') if r.strip()]
            
            with st.spinner(f"Analisando {len(reviews)} reviews..."):
                resultados_lote = []
                for review in reviews:
                    limpo = clean_text_basic(review)
                    vec = tfidf.transform([limpo])
                    proba = modelo.predict_proba(vec)[0]
                    resultados_lote.append({
                        'Review': review[:80] + '...' if len(review) > 80 else review,
                        'Sentimento': '😊 Positivo' if proba[1] >= threshold else '😞 Negativo',
                        'P(Positivo)': f"{proba[1]*100:.1f}%",
                        'P(Negativo)': f"{proba[0]*100:.1f}%",
                        'Confiança': f"{max(proba)*100:.1f}%"
                    })
            
            import pandas as pd
            df_result = pd.DataFrame(resultados_lote)
            st.dataframe(df_result, use_container_width=True)
            
            # Resumo
            n_pos = sum(1 for r in resultados_lote if 'Positivo' in r['Sentimento'])
            n_neg = len(resultados_lote) - n_pos
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total", len(resultados_lote))
            col2.metric("Positivas", n_pos, f"{n_pos/len(resultados_lote)*100:.0f}%")
            col3.metric("Negativas", n_neg, f"-{n_neg/len(resultados_lote)*100:.0f}%")

# ============================================================
# TAB 3: SOBRE O PROJETO
# ============================================================
with tab3:
    st.subheader("📈 Sobre este projeto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Objetivo
        Automatizar a classificação de sentimentos em reviews de filmes,
        demonstrando uma aplicação completa de NLP com Machine Learning.
        
        ### 📊 Dataset
        - **IMDB Movie Reviews** (Kaggle)
        - 50.000 reviews balanceadas
        - Classes: Positivo / Negativo
        
        ### 🔧 Pipeline Técnico
        1. Limpeza de texto (HTML, stopwords, lematização)
        2. Feature Engineering (n_words, n_exclamation, etc.)
        3. Vetorização TF-IDF com bigramas
        4. Treinamento de 4 modelos (NB, LR, LinearSVC, RF)
        5. Otimização com RandomizedSearchCV
        """)
    
    with col2:
        st.markdown("""
        ### 🏆 Resultados
        """)
        if modelo_carregado:
            st.dataframe({
                'Métrica': ['F1-Macro', 'Acurácia', 'AUC-ROC'],
                'Valor': [
                    f"{metricas.get('f1_macro_test', 0):.4f}",
                    f"{metricas.get('accuracy_test', 0):.4f}",
                    f"{metricas.get('auc_roc', 0):.4f}",
                ]
            }, use_container_width=True)
        
        st.markdown("""
        ### 📚 Bibliotecas Utilizadas
        - `pandas`, `numpy` — manipulação de dados
        - `nltk` — pré-processamento de texto
        - `scikit-learn` — vetorização e modelagem
        - `matplotlib`, `seaborn` — visualizações
        - `streamlit` — deploy da aplicação
        
        ### 🔗 Links
        [GitHub Repository](#) | [Notebook EDA](#) | [LinkedIn](#)
        """)