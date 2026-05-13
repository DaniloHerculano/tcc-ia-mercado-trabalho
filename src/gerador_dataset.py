import pandas as pd
import random

ocupacoes = [
    ("2521-05", "Cientista de Dados", "Tecnologia"),
    ("2515-10", "Desenvolvedor de Software", "Tecnologia"),
    ("4110-05", "Auxiliar Administrativo", "Administrativo"),
    ("4211-25", "Caixa de Banco", "Financeiro"),
    ("7823-10", "Motorista de Caminhão", "Logística"),
    ("5134-05", "Garçom", "Serviços"),
    ("1414-10", "Gerente Comercial", "Comércio"),
    ("3511-05", "Técnico de Suporte", "Tecnologia"),
    ("5211-10", "Vendedor", "Comércio"),
    ("2124-05", "Analista RH", "Administrativo"),
]

escolaridades = ["Fundamental", "Médio", "Técnico", "Superior"]
regioes = ["Sudeste", "Sul", "Nordeste", "Norte", "Centro-Oeste"]

dados = []

for _ in range(1000):

    cbo, ocupacao, setor = random.choice(ocupacoes)

    escolaridade = random.choice(escolaridades)
    regiao = random.choice(regioes)

    renda = random.randint(1500, 15000)

    # Regras simuladas de risco
    if setor == "Administrativo":
        risco = "Alto"

    elif setor == "Tecnologia":
        risco = random.choice(["Baixo", "Médio"])

    elif renda < 3000:
        risco = "Alto"

    else:
        risco = random.choice(["Baixo", "Médio"])

    dados.append([
        cbo,
        ocupacao,
        escolaridade,
        renda,
        setor,
        regiao,
        risco
    ])

df = pd.DataFrame(dados, columns=[
    "cbo",
    "ocupacao",
    "escolaridade",
    "renda",
    "setor",
    "regiao",
    "risco_automacao"
])

df.to_csv("data/dados_ocupacoes.csv", index=False)

print("Dataset gerado com sucesso!")
print(df.head())