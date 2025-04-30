# -*- coding: utf-8 -*-
"""

### Análisis de sentimientos con SpaCy y Sklearn
+ Análisis de sentimientos
+ + DataSet: http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

##### El objetivo es clasificar las reviews en positiva o negativa
"""

import pandas as pd

# Cargamos dataset
df_yelp = pd.read_table('yelp_labelled.txt')
df_imdb = pd.read_table('imdb_labelled.txt')
df_amz = pd.read_table('amazon_cells_labelled.txt')

# Unimos los datasets
frames = [df_yelp,df_imdb,df_amz]

# Renombramos las columnas de la cabecera
for colname in frames:
    colname.columns = ["Message","Target"]

# Nombre de las columnas
for colname in frames:
    print(colname.columns)

# Asignamos claves para hacerlo más fácil
keys = ['Yelp','IMDB','Amazon']

# Concatenamos las claves y los dataframes
df = pd.concat(frames,keys=keys)

df.head()

"""###  Trabajando con SpaCy

+ Instalar SpaCy, descargar modelo e importar
"""

# instalamos librería spacy
!pip install -U pip setuptools wheel
!pip install -U spacy

# descargamos modelo español
!python -m spacy download en_core_web_sm
# !python -m spacy download es_core_news_sm

import spacy
import re

nlp = spacy.load("en_core_web_sm")

"""+ Limpiar texto"""

def clean_text(texto):
  # Eliminar caracteres especiales y símbolos
  texto_limpio = re.sub(r'[^\w\s]', '', texto)

  # Eliminar números
  texto_limpio = re.sub(r'\d+', '', texto_limpio)

  # Eliminar hashtags
  texto_limpio = re.sub(r'#\w+', '', texto_limpio)

  return texto_limpio

"""+ Normalizar texto"""

def normalize_text(texto):
  text_norm = texto.lower()

  return text_norm

"""+ Eliminar Stopwords"""

def remove_stopwords(texto):
    doc = nlp(texto)
    tokens_sin_stopwords = [token.text for token in doc if not token.is_stop]
    texto_limpio = " ".join(tokens_sin_stopwords)
    return texto_limpio

"""+ Obtener Lemma"""

def lemmatization(texto):
  doc = nlp(texto)

  # Paso 3: Eliminar stopwords, lematizar y obtener una lista de tokens
  tokens = [token.lemma_ for token in doc]

"""##### Procesamos el texto completo"""

def spacy_tokenizer(texto):
  # Paso 0: Limpiar texto
  texto = clean_text(texto)

  # Paso 1: Convertir el texto a minúsculas (normalización)
  texto = normalize_text(texto)

  # Paso 2: Procesar el texto con spaCy
  doc = nlp(texto)

  # Paso 3: Eliminar stopwords, lematizar y obtener una lista de tokens
  tokens = [token.lemma_ for token in doc if not token.is_stop]

  return tokens

"""#### Vectorizamos"""

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Vectorization
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

# Using Tfidf
tfvectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer)

"""#### Machine Learning With SKlearn"""

# paquetes ML
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# Transformación Custom con Spacy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

def clean_text(text):
    return text.strip().lower()

classifier = LinearSVC()

# Dividimos Data Set
from sklearn.model_selection import train_test_split

# Features y Labels
X = df['Message']
ylabels = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.25, random_state=42)

# Creamos el pipelien en sklearn para limpiar, tokenizar, vectorizar y clasificar
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier', classifier)])

# Fit our data
pipe.fit(X_train,y_train)

# Accuracy
print("Accuracy: ",pipe.score(X_train,y_train))

# Another random review
pipe.predict(["I recommend this movie to watch, it's great"])

example = ["I love this product so much",
 "What an inferior item! I will purchase a new one",
 "I feel happy when using your product!",
 "In my experience, the product is quite useless. After 2 weeks of use it broke."
           ]

pipe.predict(example)
