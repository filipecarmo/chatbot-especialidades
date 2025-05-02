
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

st.set_page_config(page_title="Recomendação Médica", layout="centered")

st.title("🔍 Recomendação de Especialidades Médicas")
st.write("Digite seus sintomas e veja especialidades médicas sugeridas.")

# Carregamento de dados
df = pd.read_excel("base_de_sintomas.xlsx")
sintomas = df["Sintoma (frase em linguagem natural)"].tolist()
especialidades = df["Especialidade sugerida"].tolist()

# Inicialização do modelo
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sintomas_embeddings = model.encode(sintomas, convert_to_numpy=True)

# Indexação com FAISS
sintomas_embeddings_np = np.array(sintomas_embeddings).astype('float32')
dimension = sintomas_embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(sintomas_embeddings_np)

# Inicializar histórico de sessão
if 'historico' not in st.session_state:
    st.session_state['historico'] = []

# Função de recomendação
def recomendar(texto_usuario, k=3):
    vetor = model.encode([texto_usuario]).astype('float32')
    D, I = index.search(vetor, k=k)
    return [especialidades[i] for i in I[0]]

# Entrada do usuário
entrada = st.text_input("📝 Descreva seus sintomas:", placeholder="Ex: Estou com dor de cabeça e tontura")

if entrada:
    resultados = recomendar(entrada)
    st.write("### Especialidades sugeridas:")
    for i, esp in enumerate(resultados, 1):
        st.write(f"{i}. {esp}")

    # Salvar no histórico
    st.session_state['historico'].append({
        'sintomas': entrada,
        'recomendacoes': resultados
    })

    # Avaliação
    st.markdown("### Essa sugestão foi útil?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Sim"):
            st.success("Obrigado pelo feedback!")
    with col2:
        if st.button("👎 Não"):
            st.warning("Vamos melhorar!")

    # Comentário opcional
    feedback = st.text_area("💬 Quer deixar um comentário ou sugestão?")
    if st.button("Enviar feedback"):
        st.success("Comentário registrado (simulado).")

# Histórico
if st.session_state['historico']:
    st.markdown("## 🕓 Histórico da sessão")
    for idx, item in enumerate(reversed(st.session_state['historico']), 1):
        st.markdown(f"**{idx}. Sintomas:** {item['sintomas']}")
        for esp in item['recomendacoes']:
            st.markdown(f"- {esp}")

    # Gerar CSV para download
    df_hist = pd.DataFrame([
        {'Sintomas': h['sintomas'], 'Recomendação': ", ".join(h['recomendacoes'])}
        for h in st.session_state['historico']
    ])
    csv = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar histórico (CSV)", csv, "historico.csv", "text/csv")
