import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ==========================================
# CARREGAR DADOS
# ==========================================

def carregar_dados(caminho_arquivo):

    try:

        df = pd.read_csv(
            caminho_arquivo,
            encoding="latin1"
        )

        print("\nDados carregados com sucesso!")
        print(df.head())

        return df

    except Exception as e:

        print(f"Erro ao carregar dados: {e}")

        return None


# ==========================================
# LIMPEZA
# ==========================================

def limpar_dados(df):

    print("\nRemovendo valores ausentes...")

    df = df.dropna()

    print("\nQuantidade de registros após limpeza:")
    print(df.shape)

    return df


# ==========================================
# PREPARAÇÃO
# ==========================================

def preparar_dados(df):

    df = df.dropna()

    # encoders
    le_escolaridade = LabelEncoder()
    le_setor = LabelEncoder()
    le_regiao = LabelEncoder()
    le_risco = LabelEncoder()

    # transformar categorias em números
    df["escolaridade"] = le_escolaridade.fit_transform(
        df["escolaridade"]
    )

    df["setor"] = le_setor.fit_transform(
        df["setor"]
    )

    df["regiao"] = le_regiao.fit_transform(
        df["regiao"]
    )

    df["risco_automacao"] = le_risco.fit_transform(
        df["risco_automacao"]
    )

    df["indice"] = (
        df["risco_automacao"]
        .map({
            "Baixo": 20,
            "Médio": 50,
            "Alto": 80
        })
    )

    df["crescimento"] = (
        df["renda"] / 1000
    )

    return df