import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from wordcloud import WordCloud

# Descargar recursos necesarios
nltk.download('stopwords')

# ---------- Cargar y preparar los datos ----------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("IMDB Dataset.csv")
    df = df.sample(n=200, random_state=42).reset_index(drop=True)
    return df

df = cargar_datos()

# ---------- Preprocesamiento ----------
def limpiar_texto(texto):
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = texto.lower()
    tokens = texto.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

df["cleaned_review"] = df["review"].apply(limpiar_texto)

# ---------- Modelo de análisis de sentimiento ----------
def analizar_sentimiento(texto):
    polaridad = TextBlob(texto).sentiment.polarity
    if polaridad > 0.1:
        return "Positivo"
    elif polaridad < -0.1:
        return "Negativo"
    else:
        return "Neutral"

df["sentimiento"] = df["cleaned_review"].apply(analizar_sentimiento)
df["polaridad"] = df["cleaned_review"].apply(lambda x: TextBlob(x).sentiment.polarity)

# ---------- Interfaz Web en Streamlit ----------
st.title("🎬 Análisis de Sentimientos - IMDb Reviews")

# Documentación in situ
st.markdown("## 🧾 Origen del Dataset")
st.write("""
Este conjunto de datos proviene del sitio [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
Contiene 50,000 reseñas de películas de IMDb clasificadas como positivas o negativas. Para esta aplicación, se tomó una muestra aleatoria de 200 comentarios.
""")

st.markdown("## 🧹 Preprocesamiento")
st.write("""
Se eliminó puntuación, se convirtió todo a minúsculas, se eliminaron 'stop words' y se aplicó stemming.
Luego, se utilizó **TextBlob** para analizar la polaridad del texto y clasificar cada comentario como **positivo**, **neutral** o **negativo**.
""")

# Estadísticas
st.markdown("## 📊 Estadísticas Generales")

col1, col2 = st.columns(2)
with col1:
    st.write("### Distribución de sentimientos")
    dist = df["sentimiento"].value_counts()
    st.bar_chart(dist)

with col2:
    st.write("### Porcentaje por clase")
    percent = (dist / dist.sum() * 100).round(2)
    st.write(percent.astype(str) + "%")

st.write("### Polaridad promedio general")
st.metric(label="Polaridad Promedio", value=round(df["polaridad"].mean(), 3))

# Filtro de comentarios
st.markdown("## 🔍 Filtrar Comentarios")
sent_opcion = st.selectbox("Filtrar por sentimiento:", ["Todos", "Positivo", "Neutral", "Negativo"])

if sent_opcion != "Todos":
    filtrado = df[df["sentimiento"] == sent_opcion]
else:
    filtrado = df

st.dataframe(filtrado[["review", "sentimiento", "polaridad"]])

# Nube de palabras
st.markdown("## ☁️ Nube de Palabras")

texto_completo = ' '.join(filtrado["cleaned_review"])
if texto_completo.strip():
    wc = WordCloud(width=800, height=400, background_color='white').generate(texto_completo)
    st.image(wc.to_array())
else:
    st.warning("No hay texto suficiente para generar una nube de palabras.")

# Interpretación
st.markdown("## 🔎 Interpretación y Limitaciones")
st.write("""
Los resultados indican que la mayoría de los comentarios tienden a ser positivos.
Sin embargo, este análisis usa un modelo de polaridad simple (TextBlob), por lo que puede fallar al interpretar sarcasmo, ironía o contexto.
Para mejorar los resultados, se podrían usar modelos más avanzados como VADER o BERT.
""")
