import pandas as pd
import joblib
import os

# ==========================================
# MODELO
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
# FEATURES
# ==========================================

FEATURES_MODELO = [
    "indice",
    "crescimento",
    "renda",
    "escolaridade",
    "setor",
    "regiao"
]

# ==========================================
# PREVISÃO INDIVIDUAL
# ==========================================

def prever_risco(dados):

    if modelo is None:

        return {
            "erro": "Modelo não encontrado."
        }

    df = pd.DataFrame([dados])

    df = df[FEATURES_MODELO]

    previsao = modelo.predict(df)[0]

    probabilidades = modelo.predict_proba(df)[0]

    score = round(
        max(probabilidades) * 100,
        2
    )

    mapa = {
        0: "Baixo",
        1: "Médio",
        2: "Alto"
    }

    return {
        "classe": mapa.get(
            previsao,
            previsao
        ),
        "score": score
    }

# ==========================================
# PREVISÃO EM LOTE
# ==========================================

def prever_em_lote(df):

    if modelo is None:

        return pd.DataFrame({
            "Erro": [
                "Modelo não encontrado."
            ]
        })

    df_modelo = df.copy()

    df_modelo = df_modelo[
        FEATURES_MODELO
    ]

    previsoes = modelo.predict(
        df_modelo
    )

    probabilidades = modelo.predict_proba(
        df_modelo
    )

    mapa = {
        0: "Baixo",
        1: "Médio",
        2: "Alto"
    }

    scores = []

    for prob in probabilidades:

        scores.append(
            round(max(prob) * 100, 2)
        )

    df["risco_previsto"] = [
        mapa.get(p, p)
        for p in previsoes
    ]

    df["score_risco"] = scores

    return df