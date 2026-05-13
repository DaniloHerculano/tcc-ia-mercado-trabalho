import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from previsao import prever_risco, prever_em_lote

# ==========================================
# CONFIGURAÇÃO
# ==========================================

st.set_page_config(
    page_title="IA e Mercado de Trabalho",
    layout="wide"
)

modelo = joblib.load("output/modelos/random_forest.pkl")

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
# MENU LATERAL
# ==========================================

st.sidebar.title("📌 Navegação")

pagina = st.sidebar.radio(
    "Selecione uma tela:",
    [
        "📊 Visão Geral",
        "🔍 Previsão Individual",
        "📂 Previsão em Lote",
        "📈 Feature Importance",
        "🔥 Heatmap",
        "🧠 SHAP",
        "⚔️ Comparação de Modelos",
        "🏆 Ranking",
        "ℹ️ Sobre"
    ]
)

# ==========================================
# VISÃO GERAL
# ==========================================

if pagina == "📊 Visão Geral":

    st.title("🤖 IA e Mercado de Trabalho")

    st.markdown("""
    Dashboard desenvolvido para análise do impacto da Inteligência Artificial
    sobre ocupações profissionais utilizando Machine Learning.

    Modelo utilizado:
    - Random Forest
    - Python
    - Scikit-Learn
    - Streamlit
    """)

    st.image(
        "output/graficos/distribuicao_risco.png",
        use_container_width=True
    )

# ==========================================
# PREVISÃO INDIVIDUAL
# ==========================================

elif pagina == "🔍 Previsão Individual":

    st.title("🔍 Previsão Individual")

    col1, col2, col3 = st.columns(3)

    with col1:
        indice = st.slider(
            "Índice de Automação",
            0,
            100,
            50
        )

    with col2:
        crescimento = st.slider(
            "Crescimento (%)",
            -20,
            20,
            5
        )

    with col3:
        renda = st.number_input(
            "Renda Média",
            1000,
            30000,
            5000
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        escolaridade_txt = st.selectbox(
            "Escolaridade",
            list(ESCOLARIDADE_MAP.keys())
        )

    with col5:
        setor_txt = st.selectbox(
            "Setor",
            list(SETOR_MAP.keys())
        )

    with col6:
        regiao_txt = st.selectbox(
            "Região",
            list(REGIAO_MAP.keys())
        )

    if st.button("🚀 Prever"):

        dados = {
            "indice": indice,
            "crescimento": crescimento,
            "renda": renda,
            "escolaridade": ESCOLARIDADE_MAP[escolaridade_txt],
            "setor": SETOR_MAP[setor_txt],
            "regiao": REGIAO_MAP[regiao_txt]
        }

        try:
            risco = prever_risco(dados)

            st.success(f"Risco previsto: {risco}")

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

            df = pd.read_csv(arquivo)

            st.dataframe(df.head())

            resultado = prever_em_lote(df)

            st.success("Previsão realizada com sucesso!")

            st.dataframe(resultado.head(20))

            csv = resultado.to_csv(index=False).encode("utf-8")

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

    st.title("📈 Importância das Variáveis")

    features = [
        "indice",
        "crescimento",
        "renda",
        "escolaridade",
        "setor",
        "regiao"
    ]

    importancias = modelo.feature_importances_

    # segurança
    tamanho = min(len(features), len(importancias))

    importancia = pd.DataFrame({
        "Variável": features[:tamanho],
        "Importância": importancias[:tamanho]
    })

    importancia = importancia.sort_values(
        by="Importância",
        ascending=False
    )

    fig = px.bar(
        importancia,
        x="Variável",
        y="Importância",
        text="Importância",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# HEATMAP
# ==========================================

elif pagina == "🔥 Heatmap":

    st.title("🔥 Heatmap de Correlação")

    try:

        from preprocessamento import (
            carregar_dados,
            limpar_dados
        )

        df = carregar_dados(
            "data/dados_ocupacoes.csv"
        )

        df = limpar_dados(df)

        colunas = [
            "indice",
            "crescimento",
            "renda"
        ]

        corr = df[colunas].corr()

        fig, ax = plt.subplots(figsize=(8, 5))

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

elif pagina == "🧠 SHAP":

    st.title("🧠 Explicabilidade da IA (SHAP)")

    st.markdown("""
    O gráfico SHAP mostra quais variáveis mais
    influenciam as previsões do modelo.
    """)

    caminho_shap = "output/graficos/shap_summary.png"

    if os.path.exists(caminho_shap):

        st.image(
            caminho_shap,
            caption="Resumo SHAP",
            width='stretch'
        )

    else:

        st.error(
            "Imagem SHAP não encontrada."
        )

# ==========================================
# COMPARAÇÃO DE MODELOS
# ==========================================

elif pagina == "⚔️ Comparação de Modelos":

    st.title("⚔️ Comparação de Modelos de Machine Learning")

    try:

        from preprocessamento import (
            carregar_dados,
            limpar_dados
        )

        from comparacao_modelos import comparar_modelos

        df = carregar_dados(
            "data/dados_ocupacoes.csv"
        )

        df = limpar_dados(df)

        resultados = comparar_modelos(df)

        st.dataframe(resultados)

        fig = px.bar(
            resultados,
            x="Modelo",
            y="Accuracy",
            color="Modelo",
            text="Accuracy",
            title="Comparação de Accuracy"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    except Exception as e:

        st.error(e)

# ==========================================
# RANKING
# ==========================================

elif pagina == "🏆 Ranking":

    st.title("🏆 Ranking de Risco")

    try:

        df = pd.read_csv("data/dados_ocupacoes.csv")

        ranking = (
            df["risco_automacao"]
            .value_counts()
            .reset_index()
        )

        ranking.columns = ["Risco", "Quantidade"]

        fig = px.bar(
            ranking,
            x="Risco",
            y="Quantidade",
            text="Quantidade",
            title="Distribuição de Risco"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(e)

# ==========================================
# SOBRE
# ==========================================

elif pagina == "ℹ️ Sobre":

    st.title("ℹ️ Sobre o Projeto")

    st.markdown("""
    ### TCC - Ciência de Dados

    Projeto desenvolvido para análise do impacto da Inteligência Artificial
    no mercado de trabalho utilizando técnicas de Machine Learning.

    Tecnologias:
    - Python
    - Pandas
    - Scikit-Learn
    - Random Forest
    - Streamlit
    - Plotly

    Objetivo:
    Identificar profissões com maior risco de automação.
    """)