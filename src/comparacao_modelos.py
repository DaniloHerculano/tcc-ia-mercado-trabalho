import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocessamento import preparar_dados

# ==========================================
# COMPARAÇÃO DE MODELOS
# ==========================================

def comparar_modelos(df):

    # criar pasta caso não exista
    os.makedirs(
        "output/graficos",
        exist_ok=True
    )

    # preparar dados
    df = preparar_dados(df)

    # ==========================================
    # FEATURES
    # ==========================================

    FEATURES = [
        "escolaridade",
        "renda",
        "setor",
        "regiao"
    ]

    TARGET = "risco_automacao"

    # ==========================================
    # X e Y
    # ==========================================

    X = df[FEATURES]

    y = df[TARGET]

    # ==========================================
    # DIVISÃO TREINO / TESTE
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ==========================================
    # RANDOM FOREST
    # ==========================================

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)

    pred_rf = rf.predict(X_test)

    acc_rf = accuracy_score(
        y_test,
        pred_rf
    )

    # ==========================================
    # LOGISTIC REGRESSION
    # ==========================================

    lr = LogisticRegression(
        max_iter=5000
    )

    lr.fit(X_train, y_train)

    pred_lr = lr.predict(X_test)

    acc_lr = accuracy_score(
        y_test,
        pred_lr
    )

    # ==========================================
    # RESULTADOS
    # ==========================================

    resultados = pd.DataFrame({

        "Modelo": [
            "Random Forest",
            "Logistic Regression"
        ],

        "Accuracy": [
            acc_rf,
            acc_lr
        ]
    })

    print("\nResultado dos modelos:")
    print(resultados)

    # ==========================================
    # GRÁFICO
    # ==========================================

    plt.figure(figsize=(8, 5))

    plt.bar(
        resultados["Modelo"],
        resultados["Accuracy"]
    )

    plt.ylabel("Accuracy")

    plt.title(
        "Comparação de Modelos"
    )

    plt.ylim(0, 1)

    # mostrar valores no gráfico
    for i, v in enumerate(resultados["Accuracy"]):

        plt.text(
            i,
            v + 0.02,
            f"{v:.2f}",
            ha="center"
        )

    # salvar gráfico
    plt.savefig(
        "output/graficos/comparacao_modelos.png",
        bbox_inches="tight"
    )

    plt.close()

    print("\nComparação de modelos concluída!")
    print(
        "Gráfico salvo em: "
        "output/graficos/comparacao_modelos.png"
    )

    return resultados