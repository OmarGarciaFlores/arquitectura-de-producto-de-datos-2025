
# librerias para preprocesamiento de datos
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

def cargar_datos(ruta):
    """Carga un dataset desde un archivo CSV."""
    return pd.read_csv(ruta)


def procesar_datos(df):
    """
    Preprocesa un DataFrame seleccionando variables de interés, manejando valores nulos 
    y convirtiendo variables categóricas en variables dummy.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada con datos sin procesar.

    Retorna:
    --------
    pd.DataFrame
        DataFrame procesado con valores nulos manejados y variables categóricas convertidas en dummies.

    Pasos del procesamiento:
    ------------------------
    1. Se seleccionan las variables de interés, tanto numéricas como categóricas.
    2. Se identifican las variables numéricas y categóricas del DataFrame.
    3. Se manejan los valores nulos:
       - Variables categóricas: Se reemplazan los valores nulos con "null".
       - Variables numéricas: Se reemplazan los valores nulos con la mediana de la columna.
    4. Se convierten las variables categóricas en variables dummy con `pd.get_dummies()`, eliminando 
       la primera categoría para evitar colinealidad.
    5. Se asegura que todas las columnas sean de tipo numérico (`int`).
    """

    df = df.copy()  # Crear una copia del DataFrame para evitar modificar el original

    # Lista de columnas de interés
    columnas_interes = ['SalePrice', 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
                        'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
                        '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
                        'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 
                        'OpenPorchSF'] 
                        #'MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'ExterQual', 
                        #'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'KitchenQual', 'GarageFinish']

    try:
        # Verificar qué columnas están disponibles en el DataFrame
        columnas_presentes = [col for col in columnas_interes if col in df.columns]

        # Si faltan columnas, mostrar advertencia pero continuar
        columnas_faltantes = set(columnas_interes) - set(columnas_presentes)
        if columnas_faltantes:
            print(f"Advertencia: Las siguientes columnas no están en el DataFrame y serán ignoradas: {columnas_faltantes}")

        # Filtrar solo las columnas que existen
        df = df[columnas_presentes]

        # Separar numéricas y categóricas
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        num_cols = df.select_dtypes(include=['number']).columns

        # Manejo de valores nulos
        for col in cat_cols:
            df[col] = df[col].fillna("null")  # Rellenar categóricas con "null"

        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())  # Rellenar numéricas con la mediana

        # Convertir categóricas a dummies y asegurarse de que sean numéricas
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True).astype(int)

    except Exception as e:
        print(f"Error al procesar los datos: {e}")

    return df


def dividir_datos(df):
    """
    Divide un DataFrame en conjuntos de entrenamiento y prueba.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame completo que se desea dividir.

    Retorna:
    --------
    tuple (pd.DataFrame, pd.DataFrame)
        - train : DataFrame de entrenamiento (80% de los datos).
        - test : DataFrame de prueba (20% de los datos).

    Notas:
    ------
    - La división es **estratificada aleatoriamente**, asegurando reproducibilidad con `random_state=42`.
    - El conjunto de prueba representa **el 20%** del total de los datos.
    """

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    return train, test
    

if __name__ == "__main__":
    df = cargar_datos("../data/raw.csv")
    df = procesar_datos(df)
    train, test = dividir_datos(df)
    df.to_csv("../data/prep.csv", index=False)
    train.to_csv("../data/prep_train.csv", index=False)
    test.to_csv("../data/prep_test.csv", index=False)
    print("Datos procesados y guardados en data/prep.csv")