import shap
import joblib
import matplotlib.pyplot as plt
import os

# ==========================================
# CARREGAR MODELO
# ==========================================

modelo = None

CAMINHO_MODELO = "output/modelos/random_forest.pkl"

if os.path.exists(CAMINHO_MODELO):

    modelo = joblib.load(
        CAMINHO_MODELO
    )

# ==========================================
# GERAR SHAP
# ==========================================

def gerar_shap(X):

    # evita erro quando o modelo ainda não existe
    if modelo is None:

        print("\nModelo ainda não criado.")
        print("Execute primeiro: python3 src/main.py")

        return

    try:

        explainer = shap.TreeExplainer(modelo)

        shap_values = explainer.shap_values(X)

        plt.figure()

        shap.summary_plot(
            shap_values,
            X,
            show=False
        )

        plt.tight_layout()

        os.makedirs(
            "output/graficos",
            exist_ok=True
        )

        plt.savefig(
            "output/graficos/shap_summary.png",
            bbox_inches="tight"
        )

        print("\nGráfico SHAP salvo com sucesso!")

    except Exception as e:

        print(f"\nErro ao gerar SHAP: {e}")