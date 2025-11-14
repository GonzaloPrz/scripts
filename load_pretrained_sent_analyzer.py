from pathlib import Path
import pickle

lang_arg = "es"

model_path = Path(f"/Users/gp/scripts/models/sentiment-{lang_arg}/sent_analyzer.pkl")

# Cargar el modelo y el tokenizador
with open(model_path, "rb") as f:
    sent_analyzer = pickle.load(f)

print(f"Modelo de análisis de sentimiento para '{lang_arg}' cargado desde {model_path}")

res = sent_analyzer.predict("Me encanta programar en Python!")
print("Predicción realizada con éxito.")