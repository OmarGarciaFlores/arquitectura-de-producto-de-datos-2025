import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import joblib  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore

def cargar_datos(test_path):
    """Carga los datos de prueba desde un archivo CSV."""
    return pd.read_csv(test_path)

def preparar_datos(test, target_column='SalePrice'):
    """Separa las variables independientes y la variable objetivo."""
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]
    return X_test, y_test

def cargar_modelo(filename="models/model.joblib"):
    """Carga el modelo desde un archivo .joblib."""
    return joblib.load(filename)

def hacer_predicciones(model, X_test):
    """Realiza predicciones con el modelo cargado."""
    return model.predict(X_test)

def evaluar_modelo(y_test, y_pred):
    """Evalúa el modelo con MAE y RMSE."""
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse

def main():
    test = cargar_datos("data/prep_test.csv")
    X_test, y_test = preparar_datos(test)
    
    model = cargar_modelo()
    y_pred = hacer_predicciones(model, X_test)    

    mae, rmse = evaluar_modelo(y_test, y_pred)
    
    print(f"Predicciones realizadas. MAE = {mae:.4f}, RMSE = {rmse:.4f}")

if __name__ == "__main__":
    main()
