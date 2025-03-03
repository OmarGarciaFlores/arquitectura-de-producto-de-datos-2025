
import pandas as pd # type: ignore
import numpy as np # type: ignore
from xgboost import XGBRegressor # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore
import joblib # type: ignore

# Cargar los datos limpios
train = pd.read_csv("../data/prep_train.csv")
test = pd.read_csv("../data/prep_test.csv")

# Separar variables independientes y variable objetivo
X_train = train.drop(columns=['SalePrice'])
X_test = test.drop(columns=['SalePrice'])
y_train = train['SalePrice']
y_test = test['SalePrice']

# Entrenar el modelo
model = XGBRegressor(learning_rate=0.15, max_depth=2, min_child_weight=1, n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Guardar el modelo entrenado
joblib.dump(model, "model.joblib")

print(f"Modelo entrenado y guardado en model.joblib; mae = {mae} y rmse = {rmse}")