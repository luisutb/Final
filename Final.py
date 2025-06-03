import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Descargar recursos
nltk.download("stopwords")
nltk.download("vader_lexicon")
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

# Limpieza de texto
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Cargar y preprocesar
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    df["clean_review"] = df["review"].apply(clean_text)
    return df

df = load_data()

# Clasificación con VADER
def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["clean_review"].apply(get_sentiment)

# Sidebar
st.sidebar.title("🔍 Filtros")
sentiment_filter = st.sidebar.multiselect(
    "Filtrar por sentimiento:", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"]
)
keyword = st.sidebar.text_input("Buscar palabra clave:")

filtered_df = df[df["sentiment"].isin(sentiment_filter)]
if keyword:
    filtered_df = filtered_df[filtered_df["clean_review"].str.contains(keyword.lower(), na=False)]

# Título
st.title("🎬 Análisis de Sentimientos con VADER - IMDB Reviews")
st.markdown("Este análisis utiliza el modelo **VADER (Valence Aware Dictionary)** para clasificar sentimientos en 5000 reseñas.")

# Estadísticas
st.subheader("📊 Estadísticas Generales")
counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"])
percentages = counts / len(df) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric("Total de reseñas", len(df))
    st.bar_chart(counts)
with col2:
    st.bar_chart(percentages)

# Nubes de palabras
st.subheader("☁️ Nubes de Palabras")

def generate_wordcloud(data, title):
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt.gcf())
    st.markdown(f"**{title}**")

generate_wordcloud(df["clean_review"], "Todas las Reseñas")

col1, col2, col3 = st.columns(3)
with col1:
    generate_wordcloud(df[df["sentiment"] == "Positive"]["clean_review"], "Positivas")
with col2:
    generate_wordcloud(df[df["sentiment"] == "Neutral"]["clean_review"], "Neutrales")
with col3:
    generate_wordcloud(df[df["sentiment"] == "Negative"]["clean_review"], "Negativas")

# Boxplot de longitud
st.subheader("🧠 Longitud de Reseñas por Sentimiento")
df["length"] = df["review"].apply(len)
fig, ax = plt.subplots()
sns.boxplot(data=df, x="sentiment", y="length", ax=ax, palette="Set2")
st.pyplot(fig)

# Tabla de reseñas
st.subheader("📃 Reseñas Filtradas")
st.dataframe(filtered_df[["review", "sentiment"]].reset_index(drop=True), use_container_width=True)

# Documentación
st.markdown("---")
st.markdown("## 📄 Documentación del Proyecto")
st.markdown("""
**🔹 Dataset:** IMDB Movie Reviews  
**🔹 Tamaño analizado:** 5000 registros  
**🔹 Preprocesamiento:**  
- Eliminación de HTML  
- Minúsculas  
- Eliminación de símbolos  
- Remoción de stopwords  

**🔹 Modelo:**  
- `VADER` (NLTK)  
- Regla: `compound >= 0.05 = positivo`, `<= -0.05 = negativo`, otro = neutral

**🔹 Visualizaciones:**  
- Nubes de palabras por clase  
- Boxplot de longitud  
- Filtros y tabla de reseñas  

**🔹 Limitaciones:**  
- VADER no capta ironía ni sarcasmo  
- No considera contexto como BERT (Se estuvo realizando pruebas con BERT el cual considera contextos, pero al ser un modelo mas pesado se tuvo que descartar)
""")


