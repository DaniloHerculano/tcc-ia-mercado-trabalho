import pandas as pd
import numpy as np

# ==========================================
# CARREGAR DADOS
# ==========================================

def carregar_dados(caminho_arquivo):

    df = pd.read_csv(
        caminho_arquivo,
        encoding="latin1"
    )

    print("\nDados carregados com sucesso!")
    print(df.head())

    return df

# ==========================================
# LIMPEZA
# ==========================================

def limpar_dados(df):

    print("\nRemovendo valores ausentes...")

    df = df.dropna()

    # ==========================================
    # CRIAR FEATURES ARTIFICIAIS
    # ==========================================

    np.random.seed(42)

    df["indice"] = np.random.randint(
        0,
        100,
        len(df)
    )

    df["crescimento"] = np.random.randint(
        -20,
        20,
        len(df)
    )

    # ==========================================
    # CONVERTER TEXTO PARA NÚMERO
    # ==========================================

    escolaridade_map = {
        "Fundamental": 0,
        "Médio": 1,
        "Técnico": 2,
        "Superior": 3,
        "Pós-graduação": 4
    }

    setor_map = {
        "Administrativo": 0,
        "Financeiro": 1,
        "Logística": 2,
        "Tecnologia": 3,
        "Saúde": 4,
        "Educação": 5,
        "Indústria": 6
    }

    regiao_map = {
        "Norte": 0,
        "Nordeste": 1,
        "Centro-Oeste": 2,
        "Sudeste": 3,
        "Sul": 4
    }

    df["escolaridade"] = df["escolaridade"].map(
        escolaridade_map
    )

    df["setor"] = df["setor"].map(
        setor_map
    )

    df["regiao"] = df["regiao"].map(
        regiao_map
    )

    # ==========================================
    # LIMPEZA FINAL
    # ==========================================

    df = df.dropna()

    print("\nQuantidade de registros após limpeza:")
    print(df.shape)

    return df