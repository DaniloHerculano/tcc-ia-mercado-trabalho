import os
import joblib
import pandas as pd

# ==========================================
# CARREGAR MODELO
# ==========================================

modelo = None

CAMINHO_MODELO = (
    "output/modelos/random_forest.pkl"
)

if os.path.exists(CAMINHO_MODELO):

    modelo = joblib.load(
        CAMINHO_MODELO
    )

# ==========================================
# PREVISÃO INDIVIDUAL
# ==========================================

def prever_risco(dados):

    if modelo is None:

        return {
            "erro": "Modelo não encontrado."
        }

    try:

        # ==================================
        # CRIAR FEATURES FALTANTES
        # ==================================

        risco_temp = 1

        dados["indice"] = (
            risco_temp * 40
        )

        dados["crescimento"] = (
            dados["renda"] / 1000
        )

        # ==================================
        # ORDEM DAS FEATURES
        # ==================================

        colunas = [
            "indice",
            "crescimento",
            "renda",
            "escolaridade",
            "setor",
            "regiao"
        ]

        df = pd.DataFrame(
            [dados]
        )[colunas]

        # ==================================
        # PREVISÃO
        # ==================================

        probabilidades = (
            modelo.predict_proba(df)
        )

        score = max(
            probabilidades[0]
        ) * 100

        previsao = (
            modelo.predict(df)[0]
        )

        mapa = {
            0: "Baixo",
            1: "Médio",
            2: "Alto"
        }

        risco = mapa.get(
            previsao,
            str(previsao)
        )

        return {
            "risco": risco,
            "score": round(score, 2)
        }

    except Exception as e:

        return {
            "erro": str(e)
        }

# ==========================================
# PREVISÃO EM LOTE
# ==========================================

def prever_em_lote(df):

    if modelo is None:

        raise Exception(
            "Modelo não encontrado."
        )

    previsoes = modelo.predict(df)

    mapa = {
        0: "Baixo",
        1: "Médio",
        2: "Alto"
    }

    df["risco_previsto"] = [
        mapa.get(p, p)
        for p in previsoes
    ]

    return df