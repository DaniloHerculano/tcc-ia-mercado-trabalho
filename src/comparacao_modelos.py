import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def comparar_modelos(df):

    X = df[
        [
            "indice",
            "crescimento",
            "renda",
            "escolaridade",
            "setor",
            "regiao"
        ]
    ]

    y = df["risco_automacao"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    modelos = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier()
    }

    resultados = []

    for nome, modelo in modelos.items():

        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)

        resultados.append({
            "Modelo": nome,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test,
                y_pred,
                average="weighted"
            ),
            "Recall": recall_score(
                y_test,
                y_pred,
                average="weighted"
            ),
            "F1-Score": f1_score(
                y_test,
                y_pred,
                average="weighted"
            )
        })

    return pd.DataFrame(resultados)