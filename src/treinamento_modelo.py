from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

# ==========================================
# FEATURES REAIS DO DATASET
# ==========================================

FEATURES = [
    "escolaridade",
    "renda",
    "setor",
    "regiao"
]

# ==========================================
# TREINAMENTO
# ==========================================

def treinar_modelo(df):

    # encoders
    le_escolaridade = LabelEncoder()
    le_setor = LabelEncoder()
    le_regiao = LabelEncoder()
    le_risco = LabelEncoder()

    # transformar texto em número
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

    # features
    X = df[FEATURES]

    # target
    y = df["risco_automacao"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # modelo
    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\nRelatório:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # criar pasta
    os.makedirs(
        "output/modelos",
        exist_ok=True
    )

    # salvar modelo
    joblib.dump(
        modelo,
        "output/modelos/random_forest.pkl"
    )

    print("\nModelo salvo com sucesso!")

    return modelo, X_test, y_test, y_pred, X