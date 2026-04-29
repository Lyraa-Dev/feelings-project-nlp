import re
import nltk
import numpy as np
import pandas as pd
from nltk import tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

nltk.download ('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True) 

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    from nltk.corpus import wordnet
    if treebank_tag.startswith('J'):
        return wordnet.ADJ  # Adjetivo
    elif treebank_tag.startswith('V'):
        return wordnet.VERB  # Verbo
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN  # Noun
    elif treebank_tag.startswith('R'):
        return wordnet.ADV  # Adverbio
    else:
        return wordnet.NOUN  # Default to noun
    
def remove_html_tags(text: str)-> str: # Remove tags HTML (ex: <br />)
    return re.sub(r'<[^>]+>', ' ', text)

def remove_special_characters(text: str)-> str: # Remove caracteres especiais, mantendo apenas letras e espaços
    return re.sub(r'[^a-z\s]', ' ', text)  

def normalize_whitespace(text: str)-> str: # Normaliza espaços em branco (remove múltiplos espaços)
    return re.sub(r'\s+', ' ', text).strip()

def lemmatize_text(text: str)-> str: # Lematiza o texto usando POS tagging para melhorar a precisão
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = [
        LEMMATIZER.lemmatize(token, get_wordnet_pos(tag)) 
        for token, tag in pos_tags
    ]
    return ' '.join(lemmatized_tokens) 

def clean_text_basic(text: str)-> str: # Função de limpeza básica lowercase → remove HTML → remove especiais → stopwords → normaliza
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_special_characters(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return normalize_whitespace(' '.join(tokens))

def clean_text_full(text: str)-> str: # Função de limpeza completa (basic + lemmatização)
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_special_characters(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    text = ' '.join(tokens)
    text = lemmatize_text(text)
    # Remover stopwords novamente (lematização pode gerar novas)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return normalize_whitespace(' '.join(tokens))

def extract_text_features(df): # Extrai features textuais simples (número de caracteres, palavras, exclamações, etc.)
    
    df = df.copy()
    df['feat_n_chars']       = df['review'].apply(len)
    df['feat_n_words']       = df['review'].apply(lambda x: len(x.split()))
    df['feat_n_exclamation'] = df['review'].apply(lambda x: x.count('!'))
    df['feat_n_question']    = df['review'].apply(lambda x: x.count('?'))
    df['feat_n_caps_words']  = df['review'].apply(
        lambda x: sum(1 for w in x.split() if w.isupper() and len(w) > 2)
    )
    df['feat_caps_ratio']    = df['review'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    df['feat_avg_word_len']  = df['review'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    return df


if __name__ == "__main__":
    # Teste rápido das funções
    import numpy as np
    test = "This movie was NOT good at all! The acting was terrible... <br/> Worst film EVER."
    print("Original:", test)
    print("Básica:  ", clean_text_basic(test))
    print("Completa:", clean_text_full(test))