
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore
from xgboost import XGBRegressor  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore

def cargar_datos(train_path):
    """Carga los datos de entrenamiento desde un archivo CSV."""
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

def guardar_modelo(model, filename="model.joblib"):
    """Guarda el modelo entrenado en un archivo .joblib."""
    joblib.dump(model, filename)

def main():
    train = cargar_datos("../data/prep_train.csv")
    X_train, y_train = preparar_datos(train)
    
    model = entrenar_modelo(X_train, y_train)
    guardar_modelo(model)
    
    print(f"Modelo entrenado y guardado en model.joblib")

if __name__ == "__main__":
    main()
