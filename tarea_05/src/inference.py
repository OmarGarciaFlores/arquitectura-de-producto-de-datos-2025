import sys
import os

# Agregar la carpeta raíz del proyecto al path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prep import procesar_datos  
import pandas as pd # type: ignore
import joblib # type: ignore


# Cargar datos de inferencia
df = pd.read_csv("../data/inference.csv")

df = procesar_datos(df)

# Cargar modelo entrenado
model = joblib.load("model.joblib")

# Hacer predicciones
predic = model.predict(df)

# Guardar predicciones
df["prediction"] = predic
df.to_csv("../data/predictions.csv", index=False)

print("Predicciones guardadas en carpeta data")