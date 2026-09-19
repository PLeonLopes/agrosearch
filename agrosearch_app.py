"""
AgroSearch - Motor de Busca Inteligente (Índice Invertido + TF-IDF from scratch)
=================================================================================
Laboratório Prático 04 - Desafio Integrador - UNIPÊ
Disciplina: Tendências em Ciência da Computação
Professor: Me. Ricardo Roberto de Lima
Aluno: Pedro Nícollas Pereira Leon Lopes

Execução:
    pip install -r requirements.txt
    streamlit run agrosearch_app.py

Restrição do enunciado: Proibido usar bibliotecas de alto nível (scikit-learn,
TfidfVectorizer). O índice invertido e o TF-IDF são implementados 'do zero'.

Arquitetura (3 fases + bônus):
    Fase 1 - Pipeline de pré-processamento (tokenização, normalização,
             remoção de stopwords e stemming rudimentar), com stemming e
             stopwords liga/desliga via checkbox na barra lateral.
    Fase 2 - Índice Invertido em memória (Termo -> [IDs de Docs]).
    Fase 3 - Busca e ranqueamento por TF-IDF acumulado sobre a query.
    Bônus  - Similaridade de Cosseno entre o vetor da Query e o vetor
             TF-IDF de cada documento.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

# ==========================================================================
# FASE 1 - CORPUS AGRÍCOLA e PIPELINE DE PRÉ-PROCESSAMENTO
# ==========================================================================

CORPUS_PATH = Path(__file__).parent / "data" / "corpus_agricola.json"


@st.cache_data(show_spinner=False)
def carregar_corpus(caminho: str) -> list[dict]:
    """Fase 1: carrega a base de manuais técnicos (hardcode do enunciado)."""
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)["documentos"]


CORPUS: list[dict] = carregar_corpus(str(CORPUS_PATH))
N_DOCS = len(CORPUS)

# Stopwords em português (lista curada local - sem dependências externas)
STOPWORDS_PT: set[str] = {
    "a", "o", "e", "é", "de", "do", "da", "dos", "das", "em", "um", "uma",
    "uns", "umas", "para", "com", "sem", "por", "que", "se", "na", "no",
    "nas", "nos", "ao", "aos", "à", "às", "os", "as", "ou", "mais", "menos",
    "muito", "pouco", "já", "também", "como", "quando", "onde", "qual",
    "quais", "este", "esta", "esse", "essa", "isso", "isto", "aquele",
    "aquela", "seu", "sua", "seus", "suas", "ser", "estar", "há", "num",
    "numa", "pelo", "pela", "pelos", "pelas", "até", "sob", "entre",
    "após", "sobre", "durante", "e",
}

# Sufixos comuns removidos pelo stemmer rudimentar (do maior para o menor,
# para não cortar um sufixo curto quando um mais específico se aplica).
_SUFIXOS_STEMMING: list[tuple[str, str]] = [
    ("acoes", ""), ("agens", "agem"), ("mente", ""),
    ("ativo", "at"), ("ativa", "at"), ("oso", ""), ("osa", ""),
    ("ador", "a"), ("adora", "a"), ("ados", "ad"), ("adas", "ad"),
    ("ando", "a"), ("endo", "e"), ("acao", ""), ("ecao", "ec"),
    ("ismo", ""), ("ista", ""), ("mento", ""),
    ("es", ""), ("as", ""), ("os", ""),  # plural simples (deixa por último)
]


def tokenizar(texto: str) -> list[str]:
    """1a etapa: separa o texto em tokens (palavras)."""
    return re.findall(r"\b\w+\b", texto)


def normalizar_token(token: str) -> str:
    """2a etapa: minúsculas e remoção de acentos (NFD -> ASCII)."""
    token = token.lower()
    token = unicodedata.normalize("NFD", token).encode("ascii", "ignore").decode("utf-8")
    return token


def stem(token: str) -> str:
    """4a etapa (rudimentar, from scratch): corta sufixos comuns do português."""
    for sufixo, troca in _SUFIXOS_STEMMING:
        if len(token) - len(sufixo) >= 3 and token.endswith(sufixo):
            return token[: -len(sufixo)] + troca
    return token


def preprocess(texto: str, usar_stopwords: bool, usar_stemming: bool) -> list[str]:
    """Pipeline completo: Tokenização -> Normalização -> Stopwords -> Stemming."""
    tokens = tokenizar(texto)
    tokens = [normalizar_token(t) for t in tokens]
    if usar_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_PT]
    if usar_stemming:
        tokens = [stem(t) for t in tokens]
    return [t for t in tokens if t]


# ==========================================================================
# FASE 2 - ÍNDICE INVERTIDO (TERMO -> [IDs de Docs])
# ==========================================================================


def construir_indice_invertido(docs_tokens: list[list[str]], doc_ids: list[str]) -> dict[str, list[str]]:
    """Constrói o índice invertido em memória a partir dos tokens já
    pré-processados de cada documento."""
    indice: dict[str, list[str]] = defaultdict(list)
    for doc_id, tokens in zip(doc_ids, docs_tokens):
        for termo in set(tokens):
            indice[termo].append(doc_id)
    return dict(sorted(indice.items()))


# ==========================================================================
# FASE 3 - TF-IDF (DO ZERO) E RANQUEAMENTO
# ==========================================================================


def calcular_tf(termo: str, tokens_doc: list[str]) -> float:
    """TF(t, d) = (nº de ocorrências de t em d) / (nº total de termos em d)."""
    if not tokens_doc:
        return 0.0
    return tokens_doc.count(termo) / len(tokens_doc)


def calcular_idf(termo: str, docs_tokens: list[list[str]]) -> float:
    """IDF(t) = log(N / DF(t)), onde DF(t) é o nº de documentos que contêm t."""
    df_t = sum(1 for tokens in docs_tokens if termo in tokens)
    if df_t == 0:
        return 0.0
    return math.log(len(docs_tokens) / df_t)


def buscar_tfidf(query_tokens: list[str], docs_tokens: list[list[str]]) -> tuple[list[dict], dict]:
    """Fase 3: para cada termo da query, calcula TF, IDF e TF-IDF em cada
    documento e acumula (soma) o TF-IDF por documento. Retorna o detalhamento
    por termo/documento e o IDF de cada termo da query (reaproveitado no bônus)."""
    idf_por_termo = {termo: calcular_idf(termo, docs_tokens) for termo in query_tokens}
    detalhamento = []
    for termo in query_tokens:
        idf = idf_por_termo[termo]
        for tokens_doc in docs_tokens:
            tf = calcular_tf(termo, tokens_doc)
            detalhamento.append({"termo": termo, "tf": tf, "idf": idf, "tfidf": tf * idf})
    return detalhamento, idf_por_termo


# ==========================================================================
# BÔNUS - SIMILARIDADE DE COSSENO (VETORES TF-IDF DO ZERO)
# ==========================================================================


def construir_vocabulario(docs_tokens: list[list[str]]) -> list[str]:
    vocab = set()
    for tokens in docs_tokens:
        vocab.update(tokens)
    return sorted(vocab)


def vetor_tfidf_documento(tokens_doc: list[str], vocabulario: list[str], docs_tokens: list[list[str]]) -> list[float]:
    return [calcular_tf(termo, tokens_doc) * calcular_idf(termo, docs_tokens) for termo in vocabulario]


def vetor_tfidf_query(query_tokens: list[str], vocabulario: list[str], docs_tokens: list[list[str]]) -> list[float]:
    return [calcular_tf(termo, query_tokens) * calcular_idf(termo, docs_tokens) for termo in vocabulario]


def similaridade_cosseno(v1: list[float], v2: list[float]) -> float:
    """cos(θ) = (v1 . v2) / (||v1|| * ||v2||) - calculado sem numpy/sklearn."""
    produto_escalar = sum(a * b for a, b in zip(v1, v2))
    norma1 = math.sqrt(sum(a * a for a in v1))
    norma2 = math.sqrt(sum(b * b for b in v2))
    if norma1 == 0 or norma2 == 0:
        return 0.0
    return produto_escalar / (norma1 * norma2)


# ==========================================================================
# INTERFACE STREAMLIT
# ==========================================================================

st.set_page_config(page_title="AgroSearch - Motor de Busca Inteligente",
                   page_icon="🌱", layout="wide")

st.title("🌱 AgroSearch - Motor de Busca Inteligente")
st.caption("Índice Invertido + TF-IDF implementados do zero - Desafio Lab UNIPÊ "
           "(AgroTech Solutions)")

# ------------------------------ Sidebar --------------------------------
st.sidebar.header("Pipeline de Pré-processamento")
usar_stopwords = st.sidebar.checkbox("Remover Stopwords", value=True)
usar_stemming = st.sidebar.checkbox("Aplicar Stemming", value=True)
st.sidebar.caption("Ligue/desligue para ver o vocabulário, o índice invertido "
                   "e o ranking TF-IDF mudarem dinamicamente.")

st.sidebar.markdown("---")
EXEMPLOS = {
    "Irrigação da soja": "irrigação soja",
    "Controle de lagartas": "lagartas controle biológico",
    "Adubação e nitrogênio": "adubação nitrogênio milho",
    "Irrigação por gotejamento": "irrigação gotejamento",
}
exemplo = st.sidebar.selectbox("Consulta de exemplo:", ["(nenhum)"] + list(EXEMPLOS.keys()))

# Pré-processa todos os documentos com as opções atuais
for doc in CORPUS:
    doc["tokens"] = preprocess(doc["texto"], usar_stopwords, usar_stemming)
DOCS_TOKENS: list[list[str]] = [d["tokens"] for d in CORPUS]
DOC_IDS: list[str] = [d["id"] for d in CORPUS]

# ------------------------------ Consulta -------------------------------
default_query = EXEMPLOS.get(exemplo, "") if exemplo != "(nenhum)" else ""
query = st.text_input("Digite sua consulta técnica:", value=default_query,
                      placeholder="ex.: irrigação soja, lagartas controle biológico...")

tab_pipeline, tab_indice, tab_busca, tab_bonus = st.tabs([
    "🧹 Fase 1 - Pipeline", "🔗 Fase 2 - Índice Invertido",
    "📊 Fase 3 - Busca TF-IDF", "🧭 Bônus - Similaridade de Cosseno",
])

# --------------------------- FASE 1 - PIPELINE ---------------------------
with tab_pipeline:
    st.subheader("Pipeline de Pré-processamento (4 etapas)")
    doc_demo_id = st.selectbox(
        "Documento para demonstrar o pipeline:", DOC_IDS,
        format_func=lambda doc_id: f"{doc_id} - {next(d['titulo'] for d in CORPUS if d['id'] == doc_id)}",
    )
    texto_demo = next(d["texto"] for d in CORPUS if d["id"] == doc_demo_id)

    tokens_brutos = tokenizar(texto_demo)
    tokens_norm = [normalizar_token(t) for t in tokens_brutos]
    tokens_sem_stop = [t for t in tokens_norm if t not in STOPWORDS_PT] if usar_stopwords else tokens_norm
    tokens_finais = [stem(t) for t in tokens_sem_stop] if usar_stemming else tokens_sem_stop

    st.write(f"**Texto original:** {texto_demo}")
    with st.expander("1. Tokenização"):
        st.write(tokens_brutos)
    with st.expander("2. Normalização (minúsculas e sem acentos)"):
        st.write(tokens_norm)
    with st.expander(f"3. Remoção de Stopwords ({'ativa' if usar_stopwords else 'desativada'})"):
        st.write(tokens_sem_stop)
    with st.expander(f"4. Stemming ({'ativo' if usar_stemming else 'desativado'})"):
        st.write(tokens_finais)

    st.markdown("---")
    st.subheader("Vocabulário do Corpus (todos os documentos)")
    vocabulario_atual = construir_vocabulario(DOCS_TOKENS)
    st.write(f"**{len(vocabulario_atual)} termos únicos** com as opções atuais "
             f"(stopwords={'on' if usar_stopwords else 'off'}, "
             f"stemming={'on' if usar_stemming else 'off'}):")
    st.write(sorted(vocabulario_atual))

# ------------------------ FASE 2 - ÍNDICE INVERTIDO -----------------------
with tab_indice:
    st.subheader("Índice Invertido (Termo -> [IDs de Documentos])")
    indice_invertido = construir_indice_invertido(DOCS_TOKENS, DOC_IDS)
    st.caption(f"{len(indice_invertido)} termos indexados a partir dos tokens pré-processados.")

    modo_exibicao = st.radio("Formato de exibição:", ["Tabela", "JSON"], horizontal=True)
    if modo_exibicao == "Tabela":
        df_indice = pd.DataFrame([
            {"Termo": termo, "Documentos": ", ".join(docs), "DF (nº docs)": len(docs)}
            for termo, docs in indice_invertido.items()
        ])
        st.dataframe(df_indice, use_container_width=True, hide_index=True)
    else:
        st.json(indice_invertido)

    with st.expander("📚 Base de Documentos (Corpus Completo)"):
        st.dataframe(pd.DataFrame(CORPUS)[["id", "titulo", "texto"]],
                     use_container_width=True, hide_index=True)

# ------------------------- FASE 3 - BUSCA TF-IDF --------------------------
with tab_busca:
    st.subheader("Busca e Ranqueamento por TF-IDF")
    if not query.strip():
        st.info("Digite uma consulta acima (ou escolha um exemplo na barra lateral) "
                "para calcular o ranking TF-IDF.")
    else:
        query_tokens = preprocess(query, usar_stopwords, usar_stemming)
        st.write(f"**Tokens da query após o pipeline:** `{query_tokens}`")

        if not query_tokens:
            st.warning("Todos os termos da consulta foram removidos pelo pipeline "
                       "(ex.: eram apenas stopwords). Tente outra consulta.")
        else:
            detalhamento, idf_por_termo = buscar_tfidf(query_tokens, DOCS_TOKENS)

            linhas_ranking = []
            for doc_id, tokens_doc in zip(DOC_IDS, DOCS_TOKENS):
                tfidf_acumulado = sum(calcular_tf(t, tokens_doc) * idf_por_termo[t] for t in query_tokens)
                linhas_ranking.append({"ID": doc_id, "TF-IDF Acumulado": round(tfidf_acumulado, 5)})
            df_ranking = pd.DataFrame(linhas_ranking).sort_values(
                "TF-IDF Acumulado", ascending=False).reset_index(drop=True)
            df_ranking = df_ranking.merge(
                pd.DataFrame(CORPUS)[["id", "titulo", "texto"]].rename(columns={"id": "ID"}),
                on="ID", how="left")
            df_ranking = df_ranking[["ID", "titulo", "texto", "TF-IDF Acumulado"]]

            def _destacar_vencedor(row):
                cor = "background-color: rgba(76, 175, 80, 0.25)" if row.name == 0 else ""
                return [cor] * len(row)

            st.dataframe(df_ranking.style.apply(_destacar_vencedor, axis=1),
                         use_container_width=True, hide_index=True)

            if df_ranking.iloc[0]["TF-IDF Acumulado"] > 0:
                vencedor = df_ranking.iloc[0]
                st.success(f"🏆 Documento mais relevante: **{vencedor['ID']}** - "
                          f"{vencedor['titulo']} (TF-IDF acumulado = "
                          f"{vencedor['TF-IDF Acumulado']})")
            else:
                st.warning("Nenhum documento contém os termos da consulta.")

            with st.expander("🔍 Detalhamento do cálculo (TF, IDF e TF-IDF por termo/documento)"):
                df_detalhe = pd.DataFrame(detalhamento)
                df_detalhe["Documento"] = DOC_IDS * len(query_tokens)
                df_detalhe = df_detalhe[["Documento", "termo", "tf", "idf", "tfidf"]]
                df_detalhe.columns = ["Documento", "Termo", "TF", "IDF", "TF-IDF"]
                df_detalhe[["TF", "IDF", "TF-IDF"]] = df_detalhe[["TF", "IDF", "TF-IDF"]].round(5)
                st.dataframe(df_detalhe, use_container_width=True, hide_index=True)
                st.latex(r"TF(t,d) = \frac{\text{ocorrências de } t \text{ em } d}"
                         r"{\text{total de termos em } d} \qquad "
                         r"IDF(t) = \log\left(\frac{N}{DF(t)}\right)")
                st.caption("TF-IDF acumulado(d) = Σ TF(t, d) × IDF(t), para cada termo t da query.")

# ---------------------- BÔNUS - SIMILARIDADE DE COSSENO --------------------
with tab_bonus:
    st.subheader("Bônus: Similaridade de Cosseno (Query × Documentos)")
    st.caption("Compara o vetor TF-IDF completo da query com o vetor TF-IDF "
               "completo de cada documento, no espaço de todo o vocabulário - "
               "lida melhor com consultas de múltiplas palavras que o TF-IDF "
               "acumulado simples.")
    if not query.strip():
        st.info("Digite uma consulta na parte superior da página para calcular "
                "a similaridade de cosseno.")
    else:
        query_tokens = preprocess(query, usar_stopwords, usar_stemming)
        if not query_tokens:
            st.warning("Todos os termos da consulta foram removidos pelo pipeline.")
        else:
            vocabulario = construir_vocabulario(DOCS_TOKENS)
            vetor_query = vetor_tfidf_query(query_tokens, vocabulario, DOCS_TOKENS)

            linhas_cos = []
            for doc_id, tokens_doc in zip(DOC_IDS, DOCS_TOKENS):
                vetor_doc = vetor_tfidf_documento(tokens_doc, vocabulario, DOCS_TOKENS)
                cos = similaridade_cosseno(vetor_query, vetor_doc)
                linhas_cos.append({"ID": doc_id, "Similaridade de Cosseno": round(cos, 5)})

            df_cos = pd.DataFrame(linhas_cos).sort_values(
                "Similaridade de Cosseno", ascending=False).reset_index(drop=True)
            df_cos = df_cos.merge(
                pd.DataFrame(CORPUS)[["id", "titulo"]].rename(columns={"id": "ID"}),
                on="ID", how="left")
            df_cos = df_cos[["ID", "titulo", "Similaridade de Cosseno"]]

            def _destacar_vencedor_cos(row):
                cor = "background-color: rgba(33, 150, 243, 0.25)" if row.name == 0 else ""
                return [cor] * len(row)

            st.dataframe(df_cos.style.apply(_destacar_vencedor_cos, axis=1),
                         use_container_width=True, hide_index=True)

            if df_cos.iloc[0]["Similaridade de Cosseno"] > 0:
                vencedor_cos = df_cos.iloc[0]
                st.success(f"🏆 Mais similar por cosseno: **{vencedor_cos['ID']}** - "
                          f"{vencedor_cos['titulo']} (cos θ = "
                          f"{vencedor_cos['Similaridade de Cosseno']})")
            st.latex(r"\cos(\theta) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \, \|\vec{d}\|}")
            st.caption(f"Vocabulário completo usado nos vetores: {len(vocabulario)} termos.")
