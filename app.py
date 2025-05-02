import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

st.set_page_config(page_title="Recomendação Médica", layout="centered")

st.title("🔍 Recomendação de Especialidades Médicas")
st.write("Digite seus sintomas em linguagem natural e veja as especialidades sugeridas.")

# Carregar base e modelo
df = pd.read_excel("base_de_sintomas.xlsx")
sintomas = df["Sintoma (frase em linguagem natural)"].tolist()
especialidades = df["Especialidade sugerida"].tolist()

# Carregar modelo e gerar embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sintomas_embeddings = model.encode(sintomas, convert_to_numpy=True)

# Criar índice FAISS
sintomas_embeddings_np = np.array(sintomas_embeddings).astype('float32')
dimension = sintomas_embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(sintomas_embeddings_np)

# Função de recomendação
def recomendar(texto_usuario, k=3):
    vetor = model.encode([texto_usuario]).astype('float32')
    D, I = index.search(vetor, k=k)
    return [especialidades[i] for i in I[0]]

# Interface Streamlit
entrada = st.text_input("📝 Descreva seus sintomas:", placeholder="Ex: Estou com dor no peito e falta de ar")

if entrada:
    resultados = recomendar(entrada)
    st.write("### Especialidades sugeridas:")
    for i, esp in enumerate(resultados, 1):
        st.write(f"{i}. {esp}")
