# 🧠 Análisis de Sentimientos con SpaCy y Scikit-learn

Este proyecto utiliza procesamiento de lenguaje natural (NLP) con **SpaCy** para el preprocesamiento de texto y **Scikit-learn** para la clasificación de sentimientos de frases en reseñas de Yelp, IMDB y Amazon.

---

## 📦 Dataset

Se utiliza el dataset **Sentiment Labelled Sentences** de UCI:

- [http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences](http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences)

Incluye 3 archivos:

- `yelp_labelled.txt`
- `imdb_labelled.txt`
- `amazon_cells_labelled.txt`

---

## 📌 Objetivo

Clasificar frases como **positivas (1)** o **negativas (0)** mediante modelos de Machine Learning con un pipeline de limpieza, lematización, vectorización y clasificación.

---

## 🧪 Flujo del Proyecto

1. **Carga y combinación de datasets**
2. **Preprocesamiento de texto con SpaCy**:
   - Limpieza
   - Normalización
   - Eliminación de stopwords
   - Lematización
3. **Vectorización** con:
   - `CountVectorizer`
   - `TfidfVectorizer` (opcional)
4. **Clasificación** usando:
   - `LinearSVC`
5. **Pipeline completo en Scikit-learn**
6. **Evaluación y predicción de ejemplos personalizados**

---

## ▶️ Ejecución

1. Clona el repositorio y colócate en el directorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ````

3. Ejecuta el script principal:

````bash
python sentiment_spacy_sklearn.py
````

## 📈 Ejemplo de predicciones
````python
pipe.predict([
    "I love this product so much",
    "In my experience, the product is quite useless."
])
````

## 📚 Requisitos
  . Python 3.8 o superior

  . SpaCy

  . Scikit-learn

  . pandas

## 📁 Estructura de Archivos


├── sentiment_spacy_sklearn.py

├── yelp_labelled.txt

├── imdb_labelled.txt

├── amazon_cells_labelled.txt

├── requirements.txt

├── .gitignore

└── README.md
