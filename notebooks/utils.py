import os
from functools import reduce
from collections import defaultdict, Counter
import polars as pl
import pandas as pd
pl.Config(tbl_rows=50)
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel


def read_csv_with_lowercase_columns(file_path: str) -> pl.DataFrame:
    """
    Lee un archivo CSV y convierte todos los nombres de las columnas a minúsculas.

    Parameters:
    file_path (str): La ruta del archivo CSV.

    Returns:
    pl.DataFrame: Un DataFrame de polars con nombres de columnas en minúsculas.
    """
    # Leer el archivo CSV
    df = pl.read_csv(file_path,new_columns=["id"])
    
    # Transformar los nombres de las columnas a minúsculas
    df = df.rename({col: col.lower() for col in df.columns})
    
    return df

def transform_to_date(df: pl.DataFrame, columns: list) -> pl.DataFrame:
    """
    Transforms specified columns of a DataFrame into date format (YYYY-mm-dd).
    
    Parameters:
    df (pl.DataFrame): The original Polars DataFrame.
    columns (list): List of column names to transform into date format.
    
    Returns:
    pl.DataFrame: The DataFrame with specified columns transformed to dates.
    """
    transformations = []
    for col in columns:
        transformations.append(
            pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d").alias(col)
        )
    return df.with_columns(transformations)


def group_aggregate_sum(df: pl.DataFrame, group_by_cols: list, list_col: str, sum_cols: list) -> pl.DataFrame:
    """
    Groups a DataFrame by specified columns, aggregates one column's values into a list,
    and sums the specified numeric columns.
    
    Parameters:
    df (pl.DataFrame): The original Polars DataFrame.
    group_by_cols (list): List of column names to group by.
    list_col (str): The column whose values should be aggregated into a list.
    sum_cols (list): List of numeric column names on which to perform a sum.
    
    Returns:
    pl.DataFrame: The resulting grouped and aggregated Polars DataFrame.
    """
    # Group by the specified columns
    grouped_df = df.groupby(group_by_cols).agg([
        pl.col(list_col).list().alias(f"{list_col}_list"),
        *[pl.col(col).sum().alias(f"{col}_sum") for col in sum_cols]
    ])
    
    return grouped_df

def count_unique_values(df: pl.DataFrame, columns: list) -> dict:
    """
    Cuenta los valores únicos de las columnas especificadas en un DataFrame.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars.
    columns (list): Una lista de nombres de columnas para contar los valores únicos.

    Returns:
    dict: Un diccionario con los nombres de las columnas y la cantidad de valores únicos.
    """
    unique_counts = {col: df[col].n_unique() for col in columns}
    return unique_counts

def count_distinct_grouped_by(df: pl.DataFrame, y_column: str, x_columns: list, z_columns: list = None) -> pl.DataFrame:
    """
    Cuenta la cantidad de elementos distintos de una columna Y agrupado por los valores de una o más columnas X,
    ordena los resultados por una o más columnas Z de forma descendente, y calcula el porcentaje sobre el total
    de cada valor de la primera columna de agrupación.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars.
    y_column (str): El nombre de la columna Y cuyos valores distintos se contarán.
    x_columns (list): Una lista de nombres de columnas X por las que se agrupará.
    z_columns (list, optional): Una lista de nombres de columnas Z por las que se ordenará de forma descendente.

    Returns:
    pl.DataFrame: Un nuevo DataFrame con los resultados de la cantidad de valores distintos de Y agrupados por X,
                  ordenados por las columnas Z y con el porcentaje calculado.
    """
    # Agrupar por las columnas X y contar los valores distintos de Y
    result = df.group_by(x_columns).agg([
        pl.col(y_column).n_unique().alias(f"{y_column}_distinct_count")
    ])
    
    # Si hay más de una columna en x_columns, calcular el total por cada valor de la primera columna de agrupación
    if len(x_columns) > 1:
        total_by_first_column = result.group_by(x_columns[0]).agg([
            pl.col(f"{y_column}_distinct_count").sum().alias("total_first_column")
        ])
        
        # Unir los totales al resultado original
        result = result.join(total_by_first_column, on=x_columns[0])
    else:
        # Si solo hay una columna de agrupación, calcular el total general
        total_general = result[f"{y_column}_distinct_count"].sum()
        result = result.with_columns([
            pl.lit(total_general).alias("total_first_column")
        ])
    
    # Calcular el porcentaje
    result = result.with_columns([
        (pl.col(f"{y_column}_distinct_count") / pl.col("total_first_column") * 100).alias(f"{y_column}_percentage")
    ])
    
    # Si se proporcionan columnas Z, ordenar por esas columnas en orden descendente
    if z_columns:
        result = result.sort(by=z_columns, descending = True)
    
    return result

def plot_histograms(df, columns):
    """
    Plots histograms for the specified columns in the DataFrame.

    Parameters:
    df (pl.DataFrame): The DataFrame containing the data.
    columns (list): A list of column names (strings) for which histograms should be plotted.
    """
    num_columns = len(columns)
    fig, axs = plt.subplots(1, num_columns, figsize=(5*num_columns, 5))
    
    if num_columns == 1:
        axs = [axs]  # Ensure axs is iterable if there's only one column

    for i, column in enumerate(columns):
        data = df.select(column).to_series()
        axs[i].hist(data, bins=10, color='skyblue', edgecolor='black')
        axs[i].set_title(f'Histogram of {column}')
        axs[i].set_xlabel(column)
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Create an item-user matrix using reduce
def update_item_user_matrix(acc, user_basket_pair):
    """
    Updates the item-user matrix with the given user-baskets pair.

    Parameters:
    acc (dict): The accumulator dictionary, where each key is an item and each value is another dictionary.
               The inner dictionary has users as keys and the count of the item in the user's baskets as values.
    user_basket_pair (tuple): A tuple containing the user ID and their baskets.

    Returns:
    dict: The updated accumulator dictionary.
    """
    user, baskets = user_basket_pair
    for basket in baskets:
        for item in basket:
            acc[item][user] += 1
    return acc

def item_to_vector(item, item_user_matrix, user_baskets):
    """
    Returns a vector representation of an item.

    Parameters:
    item (str): The item to convert to a vector.
    item_user_matrix (dict): A dictionary where each key is an item and each value is another dictionary.
                             The inner dictionary has users as keys and the count of the item in the user's baskets as values.
    user_baskets (dict): A dictionary where each key is a user and each value is a list of items in the user's baskets.

    Returns:
    list: A list of integers representing the count of the item in each user's baskets.
    """
    return [item_user_matrix[item][user] for user in user_baskets]

def replace_sd_with_null(df: pl.DataFrame, columns: list) -> pl.DataFrame:
    """
    Reemplaza todos los valores "S/D" por valores nulos en las columnas especificadas de un DataFrame de polars.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars.
    columns (list): Una lista de nombres de columnas en las que se realizará la transformación.

    Returns:
    pl.DataFrame: Un nuevo DataFrame con los valores "S/D" reemplazados por nulos en las columnas especificadas.
    """
    # Reemplazar "S/D" con None en las columnas especificadas
    df = df.with_columns([
        pl.when(pl.col(col) == "S/D").then(None).otherwise(pl.col(col)).alias(col) for col in columns
    ])
    
    return df

def check_null_values(df: pl.DataFrame) -> pl.DataFrame:
    """
    Revisa valores nulos en un DataFrame de polars y retorna un DataFrame con la cantidad de nulos y el porcentaje de nulos por columna.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars.

    Returns:
    pl.DataFrame: Un DataFrame con tres columnas: el nombre de la columna original, el conteo de valores nulos y el porcentaje de valores nulos en esa columna.
    """
    # Calcular la cantidad de nulos por columna
    null_counts = df.select([
        pl.col(col).is_null().sum().alias(col) for col in df.columns
    ])
    
    # Calcular el total de filas en el DataFrame
    total_rows = df.height
    
    # Transponer y renombrar columnas
    result = null_counts.transpose(include_header=True, header_name="column_name").rename({"column_name": "column", "column_0": "null_count"})
    
    # Calcular el porcentaje de nulos
    result = result.with_columns([
        (pl.col("null_count") / total_rows * 100).alias("null_percentage")
    ])
    
    return result

def fill_missing_values(df: pl.DataFrame, target_column: str, numeric_features: list, categorical_features: list) -> tuple:
    """
    Rellena los valores nulos en la columna objetivo utilizando un modelo de árbol de decisión,
    realizando validación cruzada y devolviendo la evaluación del modelo.
    
    Parameters:
    df (pl.DataFrame): El DataFrame original con posibles valores nulos en la columna objetivo.
    target_column (str): El nombre de la columna objetivo donde se encuentran los valores nulos a predecir.
    numeric_features (list): Listado de nombres de variables numéricas.
    categorical_features (list): Listado de nombres de variables categóricas.
    
    Returns:
    tuple: Un DataFrame actualizado con los valores nulos rellenados, el informe de clasificación, y la precisión del modelo.
    """
    # Convertir el DataFrame de Polars a Pandas
    df_pd = df.to_pandas()

    # Codificar las variables categóricas usando OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Ajustar el encoder con todas las categorías posibles
    encoder.fit(df_pd[categorical_features])

    # Aplicar la codificación a todo el DataFrame (para el entrenamiento y la predicción)
    categorical_encoded = encoder.transform(df_pd[categorical_features])

    # Crear un DataFrame con las variables categóricas codificadas
    categorical_encoded_df = pd.DataFrame(
        categorical_encoded, 
        columns=encoder.get_feature_names_out(categorical_features)
    )

    # Combinar las variables numéricas con las variables categóricas codificadas
    X_full = pd.concat([df_pd[numeric_features], categorical_encoded_df], axis=1)

    # Identificar las filas con valores nulos en la columna objetivo
    null_rows = df_pd[df_pd[target_column].isnull()]

    # Filas que no son nulas para entrenamiento y evaluación
    X_not_null = X_full.loc[df_pd[target_column].notnull()]
    y_not_null = df_pd.loc[df_pd[target_column].notnull(), target_column]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_not_null, y_not_null, test_size=0.2, random_state=42)

    # Crear el clasificador de árbol de decisión
    tree_clf = RandomForestClassifier(random_state=42)

    # Validación cruzada
    cross_val_scores = cross_val_score(tree_clf, X_train, y_train, cv=5)

    # Entrenar el modelo
    tree_clf.fit(X_train, y_train)
    
    # Evaluación en el conjunto de prueba
    y_pred = tree_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cf = classification_report(y_test, y_pred)

    # Predicción para las filas con valores nulos
    if not null_rows.empty:
        # Extraer las filas correspondientes a los valores nulos de la columna objetivo
        X_nulls = X_full.loc[null_rows.index]

        # Usar el modelo entrenado para predecir los valores nulos
        predicted_values = tree_clf.predict(X_nulls)

        # Rellenar los valores nulos en la columna objetivo con las predicciones
        df_pd.loc[null_rows.index, target_column] = predicted_values

    # Convertir de vuelta a Polars
    df_updated = pl.from_pandas(df_pd)

    return df_updated, cf, accuracy, cross_val_scores.mean()


def filter_rows_with_nulls(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filtra las filas de un DataFrame de polars que contienen al menos un valor nulo.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars a ser filtrado.

    Returns:
    pl.DataFrame: Un nuevo DataFrame que contiene solo las filas con al menos un valor nulo.
    """
    # Crear una máscara booleana que verifica si hay algún valor nulo en cada fila
    null_mask = df.select([
        pl.col(col).is_null() for col in df.columns
    ]).with_columns([
        pl.any_horizontal("*").alias("has_null")
    ]).select("has_null")
    
    # Filtrar las filas que tienen algún valor nulo
    df_with_nulls = df.with_columns(null_mask).filter(pl.col("has_null"))
    
    return df_with_nulls

def filter_rows_without_nulls(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filtra las filas de un DataFrame de polars que no contienen ningún valor nulo.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars a ser filtrado.

    Returns:
    pl.DataFrame: Un nuevo DataFrame que contiene solo las filas sin ningún valor nulo.
    """
    # Crear una máscara booleana que verifica si no hay ningún valor nulo en cada fila
    not_null_mask = df.select([
        pl.col(col).is_not_null() for col in df.columns
    ]).with_columns([
        pl.all_horizontal("*").alias("has_no_null")
    ]).select("has_no_null")
    
    # Filtrar las filas que no tienen valores nulos
    df_without_nulls = df.filter(not_null_mask["has_no_null"])
    
    return df_without_nulls


def count_distinct_in_bins(df: pl.DataFrame, volume_column: str, column: str, bin_size: int = 500) -> pl.DataFrame:
    """
    Cuenta los valores distintos de una columna dentro de rangos de tamaño específico en otra columna.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars.
    volume_column (str): El nombre de la columna que contiene los volúmenes (por ejemplo, 'totalvolumen').
    pocr_column (str): El nombre de la columna cuyos valores distintos se contarán (por ejemplo, 'pocr').
    bin_size (int): El tamaño de cada bloque o rango (por defecto 500).

    Returns:
    pl.DataFrame: Un DataFrame con los rangos y la cantidad de valores distintos de pocr en cada rango.
    """
    # Crear una nueva columna que agrupa los valores en bloques de 'bin_size'
    df = df.with_columns([
        (pl.col(volume_column) // bin_size * bin_size).alias("volume_bin")
    ])
    
    # Agrupar por los bloques y contar los valores distintos de 'pocr'
    result = df.group_by("volume_bin").agg([
        pl.col(column).n_unique().alias(f"{column}_distinct_count")
    ])
     
     # Retornar el resultado por 'volume_bin' ordenado de menor a mayor
    return result.sort("volume_bin")

def group_and_describe_with_percentiles(df: pl.DataFrame, group_by_col: str, numeric_cols: list, percentiles: list = [0.25, 0.5, 0.75]) -> pl.DataFrame:
    """
    Agrupa el DataFrame por una columna categórica y calcula las medidas de tendencia central
    y los percentiles para una lista de variables numéricas.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars a ser agrupado y analizado.
    group_by_col (str): El nombre de la columna categórica por la que se agrupará.
    numeric_cols (list): Una lista de nombres de columnas numéricas para calcular las medidas de tendencia central.
    percentiles (list): Una lista de percentiles a calcular (por defecto [0.25, 0.5, 0.75]).

    Returns:
    pl.DataFrame: Un nuevo DataFrame con las medidas de tendencia central y percentiles para cada grupo.
    """
    # Agrupar por la columna especificada y calcular las estadísticas descriptivas para las columnas numéricas
    result = df.group_by(group_by_col).agg([
        pl.col(col).mean().alias(f"{col}_mean") for col in numeric_cols
    ] + [
        pl.col(col).median().alias(f"{col}_median") for col in numeric_cols
    ] + [
        pl.col(col).std().alias(f"{col}_std_dev") for col in numeric_cols
    ] + [
        pl.col(col).quantile(p).alias(f"{col}_p{int(p*100)}") for col in numeric_cols for p in percentiles
    ])

    return result

def plot_boxplots(df: pl.DataFrame, columns: list):
    """
    Genera un boxplot para cada columna en la lista de columnas ingresada.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars.
    columns (list): Una lista de nombres de columnas para las cuales se generarán los boxplots.
    """
    # Convertir las columnas a pandas para la compatibilidad con matplotlib
    df_pandas = df.select(columns).to_pandas()
    
    # Generar un boxplot para cada columna
    for col in columns:
        plt.figure(figsize=(8, 6))
        plt.boxplot(df_pandas[col].dropna(), vert=False, patch_artist=True)
        plt.title(f'Boxplot de {col}')
        plt.xlabel(col)
        plt.show()
        
def chi2_test(df: pl.DataFrame, col1: str, col2: str, confidence_level: float = 0.95):
    """
    Realiza la prueba de Chi-cuadrado para dos columnas categóricas y devuelve el p-valor y la conclusión.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars.
    col1 (str): El nombre de la primera columna categórica.
    col2 (str): El nombre de la segunda columna categórica.
    confidence_level (float): El grado de confianza para la prueba (por defecto 0.95).

    Returns:
    tuple: Un tuple con el p-valor y la conclusión de la prueba.
    """
    # Convertir las columnas seleccionadas a pandas
    df_pandas = df.select([col1, col2]).to_pandas()
    
    # Crear una tabla de contingencia
    contingency_table = pd.crosstab(df_pandas[col1], df_pandas[col2])
    
    # Realizar la prueba Chi-cuadrado
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Calcular el alfa basado en el grado de confianza
    alpha = 1 - confidence_level
    
    # Conclusión basada en el p-valor y el grado de confianza
    if p_value < alpha:
        conclusion = "Rechazamos la hipótesis nula: Existe una asociación significativa entre las variables."
    else:
        conclusion = "No se puede rechazar la hipótesis nula: No existe evidencia suficiente para afirmar una asociación significativa entre las variables."
    
    return p_value, conclusion

def chi2_matrix(chi2_test_func, df: pl.DataFrame, categorical_columns: list, confidence_level: float ) -> np.ndarray:
    """
    Realiza la prueba de Chi-cuadrado entre todas las combinaciones de variables categóricas y devuelve una matriz de Verdadero/Falso.

    Parameters:
    chi2_test_func (function): La función que realiza la prueba de Chi-cuadrado.
    df (pl.DataFrame): El DataFrame de polars.
    categorical_columns (list): Una lista con los nombres de las variables categóricas.
    confidence_level (float): El grado de confianza para la prueba (por defecto 0.95).

    Returns:
    np.ndarray: Una matriz con valores Verdadero/Falso indicando si se rechaza la hipótesis nula para cada combinación.
    """
    n = len(categorical_columns)
    results_matrix = np.zeros((n, n), dtype=bool)

    # Realizar la prueba Chi-2 para todas las combinaciones de columnas categóricas
    for i in range(n):
        for j in range(i + 1, n):  # Comienza desde i+1 para evitar comparaciones duplicadas
            _, conclusion = chi2_test_func(df, categorical_columns[i], categorical_columns[j], confidence_level)
            results_matrix[i, j] = conclusion
            results_matrix[j, i] = conclusion  # La matriz es simétrica
            
    # Convertir la matriz en un DataFrame de pandas y asignar nombres a filas y columnas
    results_df = pd.DataFrame(results_matrix, index=categorical_columns, columns=categorical_columns)

    return results_df


def fit_ordered_logistic_regression(df: pl.DataFrame, target_column: str, numerical_features: list):
    """
    Ajusta un modelo de Regresión Logística Ordenada utilizando variables numéricas y una variable categórica ordinal como objetivo.
    
    Parameters:
    df (pl.DataFrame): El DataFrame original en formato Polars.
    target_column (str): El nombre de la columna objetivo que es categórica y ordinal.
    numerical_features (list): Listado de nombres de variables numéricas.
    
    Returns:
    result: El resultado del ajuste del modelo de Regresión Logística Ordenada.
    """
    # Convertir el DataFrame de Polars a Pandas para usarlo con statsmodels
    df_pd = df.to_pandas()

    # Codificar la variable categórica `target_column` como ordinal
    label_encoder = LabelEncoder()
    df_pd['encoded'] = label_encoder.fit_transform(df_pd[target_column])

    # Asegurarse de que la variable objetivo es tratada como una categoría ordenada
    df_pd['encoded'] = pd.Categorical(df_pd['encoded'], ordered=True)

    # Definir las características y la variable objetivo
    X = df_pd[numerical_features]
    y = df_pd['encoded']

    # Ajustar el modelo de Regresión Logística Ordenada
    model = OrderedModel(y, X, distr='logit')
    result = model.fit(method='bfgs')

    # Mostrar los resultados
    return result
