import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

from previsao import (
    prever_risco,
    prever_em_lote
)

from main import main

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="IA e Mercado de Trabalho",
    layout="wide"
)

# ==========================================
# TREINAR MODELO SE NÃO EXISTIR
# ==========================================

if not os.path.exists(
    "output/modelos/random_forest.pkl"
):

    main()

modelo = joblib.load(
    "output/modelos/random_forest.pkl"
)

# ==========================================
# DADOS
# ==========================================

df = pd.read_csv(
    "data/dados_ocupacoes.csv",
    encoding="latin1"
)

df["risco_automacao"] = (
    df["risco_automacao"]
    .astype(str)
    .str.strip()
    .replace({
        "MÃ©dio": "Médio",
        "Medio": "Médio"
    })
)

df["setor"] = (
    df["setor"]
    .astype(str)
    .str.strip()
)

df["regiao"] = (
    df["regiao"]
    .astype(str)
    .str.strip()
)

medio = len(
    df[df["risco_automacao"] == "Médio"]
)

# ==========================================
# MAPEAMENTOS
# ==========================================

ESCOLARIDADE_MAP = {
    "Fundamental": 0,
    "Médio": 1,
    "Técnico": 2,
    "Superior": 3,
    "Pós-graduação": 4
}

SETOR_MAP = {
    "Administrativo": 0,
    "Financeiro": 1,
    "Logística": 2,
    "Tecnologia": 3,
    "Saúde": 4,
    "Educação": 5,
    "Indústria": 6
}

REGIAO_MAP = {
    "Norte": 0,
    "Nordeste": 1,
    "Centro-Oeste": 2,
    "Sudeste": 3,
    "Sul": 4
}

# ==========================================
# MENU
# ==========================================

st.sidebar.title("📌 Navegação")

pagina = st.sidebar.radio(
    "Selecione:",
    [
        "📊 Dashboard",
        "🔍 Previsão Individual",
        "📂 Previsão em Lote",
        "📈 Feature Importance",
        "🔥 Heatmap",
        "📉 SHAP",
        "🏆 Ranking",
        "⚔️ Comparação de Modelos",
        "ℹ️ Sobre"
    ]
)

# ==========================================
# DASHBOARD
# ==========================================

if pagina == "📊 Dashboard":

    st.title("🤖 IA e Mercado de Trabalho")

    st.markdown("""
    Plataforma de análise do impacto da Inteligência Artificial
    nas profissões utilizando Machine Learning.
    """)

    total = len(df)

    alto = len(
        df[df["risco_automacao"] == "Alto"]
    )

    medio = len(
        df[df["risco_automacao"] == "Médio"]
    )

    baixo = len(
        df[df["risco_automacao"] == "Baixo"]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total de Profissões",
        total
    )

    col2.metric(
        "Alto Risco",
        alto
    )

    col3.metric(
        "Médio Risco",
        medio
    )

    col4.metric(
        "Baixo Risco",
        baixo
    )

    st.divider()

    # ======================================
    # FILTROS
    # ======================================

    colf1, colf2 = st.columns(2)

    with colf1:

        setor_filtro = st.selectbox(
            "Filtrar por setor",
            ["Todos"] + sorted(
                df["setor"].dropna().unique()
                .tolist()
            )
        )

    with colf2:

        regiao_filtro = st.selectbox(
            "Filtrar por região",
            ["Todas"] + sorted(
                df["regiao"].dropna().unique()
                .tolist()
            )
        )

    busca = st.text_input(
        "🔎 Buscar profissão"
    )

    # ======================================
    # FILTROS
    # ======================================

    df_filtrado = df.copy()

    if setor_filtro != "Todos":

        df_filtrado = df_filtrado[
            df_filtrado["setor"]
            == setor_filtro
        ]

    if regiao_filtro != "Todas":

        df_filtrado = df_filtrado[
            df_filtrado["regiao"]
            == regiao_filtro
        ]

    if busca:

        df_filtrado = df_filtrado[
            df_filtrado["ocupacao"]
            .str.contains(
                busca,
                case=False,
                na=False
            )
        ]

    st.divider()

    # ======================================
    # GRÁFICO
    # ======================================

    risco_setor = (
        df_filtrado.groupby(
            ["setor", "risco_automacao"]
        )
        .size()
        .reset_index(name="quantidade")
    )

    fig = px.bar(
        risco_setor,
        x="setor",
        y="quantidade",
        color="risco_automacao",
        barmode="group",
        title="Distribuição de Risco por Setor"
    )

    st.plotly_chart(
        fig,
        width='stretch'
    )

    # ======================================
    # TABELA
    # ======================================

    st.subheader(
        "📋 Dados das Ocupações"
    )

    st.dataframe(
        df_filtrado,
        width='stretch'
    )

# ==========================================
# PREVISÃO INDIVIDUAL
# ==========================================

elif pagina == "🔍 Previsão Individual":

    st.title("🔍 Previsão Individual")

    renda = st.number_input(
        "Renda Média",
        1000,
        30000,
        5000
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        escolaridade_txt = st.selectbox(
            "Escolaridade",
            list(
                ESCOLARIDADE_MAP.keys()
            )
        )

    with col2:

        setor_txt = st.selectbox(
            "Setor",
            list(
                SETOR_MAP.keys()
            )
        )

    with col3:

        regiao_txt = st.selectbox(
            "Região",
            list(
                REGIAO_MAP.keys()
            )
        )

    if st.button("🚀 Prever"):

        dados = {
            "renda": renda,
            "escolaridade":
                ESCOLARIDADE_MAP[
                    escolaridade_txt
                ],
            "setor":
                SETOR_MAP[
                    setor_txt
                ],
            "regiao":
                REGIAO_MAP[
                    regiao_txt
                ]
        }

        try:

            resultado = prever_risco(dados)

            risco = resultado["risco"]

            score = resultado["score"]

            emoji = {
                "Baixo": "🟢",
                "Médio": "🟡",
                "Alto": "🔴"
            }

            st.success(
                f"""
{emoji.get(risco)}

Risco previsto: {risco}

Score de risco: {score}%
"""
            )

            st.progress(score / 100)

        except Exception as e:

            st.error(e)

# ==========================================
# PREVISÃO EM LOTE
# ==========================================

elif pagina == "📂 Previsão em Lote":

    st.title("📂 Previsão em Lote")

    arquivo = st.file_uploader(
        "Envie um CSV",
        type=["csv"]
    )

    if arquivo is not None:

        try:

            df_upload = pd.read_csv(
                arquivo
            )

            st.dataframe(
                df_upload.head()
            )

            resultado = prever_em_lote(
                df_upload
            )

            st.success(
                "Previsão realizada!"
            )

            st.dataframe(
                resultado.head(20)
            )

            csv = resultado.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download CSV",
                csv,
                "resultado.csv",
                "text/csv"
            )

        except Exception as e:

            st.error(e)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

elif pagina == "📈 Feature Importance":

    st.title(
        "📈 Importância das Variáveis"
    )

    features = [
        "renda",
        "escolaridade",
        "setor",
        "regiao"
    ]

    importancias = (
        modelo.feature_importances_
    )

    tamanho = min(
        len(features),
        len(importancias)
    )

    importancia = pd.DataFrame({
        "Variável":
            features[:tamanho],
        "Importância":
            importancias[:tamanho]
    })

    importancia = (
        importancia.sort_values(
            by="Importância",
            ascending=False
        )
    )

    fig = px.bar(
        importancia,
        x="Variável",
        y="Importância",
        color="Importância",
        text="Importância",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        width='stretch'
    )

# ==========================================
# HEATMAP
# ==========================================

elif pagina == "🔥 Heatmap":

    st.title(
        "🔥 Heatmap de Correlação"
    )

    try:

        df_heat = df.copy()

        # ======================================
        # CONVERTER TEXTO PARA NÚMERO
        # ======================================

        df_heat["escolaridade"] = (
            df_heat["escolaridade"]
            .map(ESCOLARIDADE_MAP)
        )

        df_heat["setor"] = (
            df_heat["setor"]
            .map(SETOR_MAP)
        )

        df_heat["regiao"] = (
            df_heat["regiao"]
            .map(REGIAO_MAP)
        )

        # ======================================
        # CONVERTER RISCO
        # ======================================

        risco_map = {
            "Baixo": 0,
            "Médio": 1,
            "Alto": 2
        }

        df_heat["risco_automacao"] = (
            df_heat["risco_automacao"]
            .map(risco_map)
        )

        # ======================================
        # COLUNAS NUMÉRICAS
        # ======================================

        colunas = [
            "renda",
            "escolaridade",
            "setor",
            "regiao",
            "risco_automacao"
        ]

        corr = df_heat[colunas].corr()

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.heatmap(
            corr,
            annot=True,
            cmap="Blues",
            ax=ax
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(e)

# ==========================================
# SHAP
# ==========================================

elif pagina == "📉 SHAP":

    st.title(
        "📉 Explicabilidade SHAP"
    )

    caminho = (
        "output/graficos/"
        "shap_summary.png"
    )

    if os.path.exists(caminho):

        st.image(
            caminho,
            width='stretch'
        )

    else:

        st.warning(
            "Imagem SHAP não encontrada."
        )

# ==========================================
# RANKING
# ==========================================

elif pagina == "🏆 Ranking":

    st.title(
        "🏆 Ranking de Risco"
    )

    ranking = (
        df["risco_automacao"]
        .value_counts()
        .reset_index()
    )

    ranking.columns = [
        "Risco",
        "Quantidade"
    ]

    fig = px.pie(
        ranking,
        values="Quantidade",
        names="Risco",
        title="Distribuição de Risco"
    )

    st.plotly_chart(
        fig,
        width='stretch'
    )

# ==========================================
# COMPARAÇÃO DE MODELOS
# ==========================================

elif pagina == "⚔️ Comparação de Modelos":

    st.title("⚔️ Comparação de Modelos")

    caminho = (
        "output/graficos/comparacao_modelos.png"
    )

    if os.path.exists(caminho):

        st.image(
            caminho,
            caption="Comparação de Accuracy",
            width='stretch'
        )

    else:

        st.error(
            "Gráfico não encontrado."
        )

# ==========================================
# SOBRE
# ==========================================

elif pagina == "ℹ️ Sobre":

    st.title("ℹ️ Sobre")

    st.markdown("""
    ### TCC - Ciência de Dados

    Projeto desenvolvido para análise do impacto
    da Inteligência Artificial no mercado de trabalho
    utilizando Machine Learning.

    ### Tecnologias

    - Python
    - Pandas
    - Scikit-Learn
    - Random Forest
    - Streamlit
    - Plotly
    - SHAP

    ### Objetivo

    Identificar profissões com maior
    risco de automação.
    """)
