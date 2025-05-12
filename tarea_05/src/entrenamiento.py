import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore
import os
from xgboost import XGBRegressor  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore

# Obtener el directorio base (dos niveles arriba de este script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Definir las rutas dinámicamente
TRAIN_PATH = os.path.join(BASE_DIR, "data", "prep_train.csv")
MODEL_OUTPUT = os.path.join(BASE_DIR, "models", "model.joblib")

def cargar_datos(train_path):
    """Carga los datos de entrenamiento desde un archivo CSV."""
    print(f"Cargando datos desde: {train_path}")  # Para depuración
    return pd.read_csv(train_path)

def preparar_datos(train, target_column='SalePrice'):
    """Separa las variables independientes y la variable objetivo."""
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    return X_train, y_train

def entrenar_modelo(X_train, y_train):
    """Entrena un modelo XGBRegressor y lo devuelve."""
    model = XGBRegressor(
        learning_rate=0.1, max_depth=2, min_child_weight=1,
        n_estimators=400, random_state=42
    )
    model.fit(X_train, y_train)
    return model

def guardar_modelo(model, filename):
    """Guarda el modelo entrenado en un archivo .joblib."""
    print(f"Guardando modelo en: {filename}")  # Para depuración
    joblib.dump(model, filename)

def main():
    print(f"Directorio base del proyecto: {BASE_DIR}")  # Para depuración
    
    # Cargar los datos de entrenamiento
    train = cargar_datos(TRAIN_PATH)
    X_train, y_train = preparar_datos(train)
    
    # Entrenar el modelo
    model = entrenar_modelo(X_train, y_train)
    
    # Guardar el modelo entrenado
    guardar_modelo(model, MODEL_OUTPUT)
    
    print(f"Modelo entrenado y guardado en {MODEL_OUTPUT}")

if __name__ == "__main__":
    main()

