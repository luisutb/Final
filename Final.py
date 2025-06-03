import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# Descargar recursos necesarios de nltk
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Cargar stopwords
stop_words = set(stopwords.words("english"))

# Preprocesamiento
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)  # quitar etiquetas HTML
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # quitar símbolos y números
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    df["clean_review"] = df["review"].apply(clean_text)
    return df

df = load_data()

# Aplicar VADER
sid = SentimentIntensityAnalyzer()

def get_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["compound"] = df["clean_review"].apply(lambda x: sid.polarity_scores(x)["compound"])
df["sentiment"] = df["compound"].apply(get_sentiment)

# Sidebar: filtros
st.sidebar.title("🔍 Filtros")
sentiment_filter = st.sidebar.multiselect(
    "Filtrar por sentimiento:", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"]
)

keyword = st.sidebar.text_input("Buscar palabra clave en la reseña:")

filtered_df = df[df["sentiment"].isin(sentiment_filter)]
if keyword:
    filtered_df = filtered_df[filtered_df["clean_review"].str.contains(keyword.lower(), na=False)]

# Título
st.title("🎬 Análisis de Sentimientos de Reseñas IMDB")
st.markdown("Este análisis usa VADER para clasificar reseñas como **positivas**, **negativas** o **neutrales**.")

# Estadísticas
st.subheader("📊 Estadísticas Generales")
counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"])
percentages = counts / len(df) * 100

col1, col2 = st.columns(2)
with col1:
    st.metric("Total de reseñas", len(df))
    st.write("### Conteo por categoría")
    st.bar_chart(counts)
with col2:
    st.write("### Porcentaje por categoría")
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

# Análisis adicional
st.subheader("🧠 Análisis Complementario")

df["length"] = df["review"].apply(len)
fig, ax = plt.subplots()
sns.boxplot(data=df, x="sentiment", y="length", ax=ax, palette="Set2")
ax.set_title("Distribución de Longitud de Reseñas por Sentimiento")
st.pyplot(fig)

# Tabla de reseñas
st.subheader("📃 Reseñas Filtradas")
st.dataframe(filtered_df[["review", "sentiment", "compound"]].reset_index(drop=True), use_container_width=True)

# Documentación
st.markdown("---")
st.markdown("## 📄 Documentación del Proyecto")
st.markdown("""
**🔹 Origen del dataset:**  
IMDB Movie Review Dataset - contiene reseñas de películas etiquetadas como positivas o negativas.

**🔹 Preprocesamiento realizado:**  
- Eliminación de HTML y símbolos.  
- Conversión a minúsculas.  
- Eliminación de palabras vacías (stopwords).  

**🔹 Herramienta utilizada para análisis de sentimiento:**  
- VADER SentimentIntensityAnalyzer (`nltk.sentiment.vader`).

**🔹 Visualizaciones:**  
- Gráficos de barras.  
- Nubes de palabras por sentimiento.  
- Boxplot de longitud de reseñas.

**🔹 Limitaciones:**  
- Solo se usa texto, no se considera tono o contexto profundo.  
- El tamaño reducido (200 reseñas) limita el poder estadístico.  
""")


