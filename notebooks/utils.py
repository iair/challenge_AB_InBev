import os
import polars as pl
pl.Config(tbl_rows=50)
import pandas as pd
import numpy as np
import pyarrow
import json
from functools import reduce
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import kruskal
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import Dict, Tuple, Any,List,Callable

def read_csv_with_lowercase_columns(file_path: str) -> pl.DataFrame:
    """
    Lee un archivo CSV y convierte todos los nombres de las columnas a minúsculas.

    Parameters:
    file_path (str): La ruta del archivo CSV.

    Returns:
    pl.DataFrame: Un DataFrame de polars con nombres de columnas en minúsculas.
    """
    # Leer el archivo CSV
    df = pl.read_csv(file_path)
    
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
def calculate_statistics(df: pl.DataFrame, numeric_cols: list) -> pl.DataFrame:
    """
    Calcula el promedio, desviación estándar, y percentiles 0, 25, 50, 75, 100 para un grupo de variables numéricas.

    Parameters:
    df (pl.DataFrame): El DataFrame de Polars con los datos.
    numeric_cols (list): Lista de nombres de las columnas numéricas a calcular.

    Returns:
    pl.DataFrame: Un nuevo DataFrame donde las filas son las variables y las columnas son las métricas calculadas.
    """
    # Crear una lista para almacenar las métricas
    metrics = []
    
    # Iterar sobre cada columna numérica y calcular las métricas
    for col in numeric_cols:
        metrics.extend([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std_dev"),
            pl.col(col).quantile(0).alias(f"{col}_p0"),
            pl.col(col).quantile(0.25).alias(f"{col}_p25"),
            pl.col(col).quantile(0.5).alias(f"{col}_p50"),
            pl.col(col).quantile(0.75).alias(f"{col}_p75"),
            pl.col(col).quantile(1).alias(f"{col}_p100")
        ])
    
    # Seleccionar y calcular las métricas
    stats_df = df.select(metrics)
    
    # Manual reshaping: collect all the metrics for each variable
    rows = []
    for col in numeric_cols:
        row = {
            "variable": col,
            "mean": stats_df.select(pl.col(f"{col}_mean")).item(),
            "std_dev": stats_df.select(pl.col(f"{col}_std_dev")).item(),
            "p0": stats_df.select(pl.col(f"{col}_p0")).item(),
            "p25": stats_df.select(pl.col(f"{col}_p25")).item(),
            "p50": stats_df.select(pl.col(f"{col}_p50")).item(),
            "p75": stats_df.select(pl.col(f"{col}_p75")).item(),
            "p100": stats_df.select(pl.col(f"{col}_p100")).item()
        }
        rows.append(row)
    
    # Convertir la lista de filas en un DataFrame final
    final_df = pl.DataFrame(rows)
    
    return final_df
def group_and_describe_with_percentiles(df: pl.DataFrame, group_by_col: str, numeric_cols: list, percentiles: list = [0.25, 0.5, 0.75]) -> pl.DataFrame:
    """
    Agrupa el DataFrame por una columna categórica y calcula las medidas de tendencia central,
    los percentiles y el conteo de elementos para una lista de variables numéricas.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars a ser agrupado y analizado.
    group_by_col (str): El nombre de la columna categórica por la que se agrupará.
    numeric_cols (list): Una lista de nombres de columnas numéricas para calcular las medidas de tendencia central.
    percentiles (list): Una lista de percentiles a calcular (por defecto [0.25, 0.5, 0.75]).

    Returns:
    pl.DataFrame: Un nuevo DataFrame con las medidas de tendencia central, percentiles y conteo para cada grupo.
    """
    # Agrupar por la columna especificada y calcular las estadísticas descriptivas para las columnas numéricas
    result = df.group_by(group_by_col).agg(
        [
            pl.count().alias("count")  # Contar el número de elementos en cada grupo
        ] + [
            pl.col(col).mean().alias(f"{col}_mean") for col in numeric_cols
        ] + [
            pl.col(col).median().alias(f"{col}_median") for col in numeric_cols
        ] + [
            pl.col(col).std().alias(f"{col}_std_dev") for col in numeric_cols
        ] + [
            pl.col(col).quantile(p).alias(f"{col}_p{int(p*100)}") for col in numeric_cols for p in percentiles
        ]
    )

    return result
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
def kruskal_wallis_test_multiple(df, numeric_cols, categorical_col, alpha=0.05):
    """
    Realiza el test de Kruskal-Wallis para evaluar la independencia entre varias variables categóricas y una variable numérica.

    Parameters:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    numeric_cols (list): Una lista de nombres de las columnas numéricas.
    categorical_col (str): El nombre de la columna categórica.
    alpha (float): Nivel de significancia para rechazar la hipótesis nula (por defecto es 0.05).

    Returns:
    dict: Un diccionario con los resultados del test para cada columna numérica.
    """
    results = {}
    
    for numeric_col in numeric_cols:
        # Agrupar los datos por la variable categórica y extraer los valores de la variable numérica
        groups = [group[numeric_col] for name, group in df.group_by(categorical_col)]
        
        # Realizar el test de Kruskal-Wallis
        stat, p_value = kruskal(*groups)
        
        # Determinar si se rechaza la hipótesis nula
        reject_null = p_value < alpha
        
        # Guardar los resultados en el diccionario
        results[numeric_col] = {
            'estadistico': stat,
            'p_valor': p_value,
            'rechaza_hipotesis_nula de independencia': reject_null
        }
    
    return results
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
def plot_boxplots(df: pl.DataFrame, columns: list):
    """
    Genera boxplots para cada columna en la lista de columnas ingresada, dispuestos en una sola fila.

    Parameters:
    df (pl.DataFrame): El DataFrame de polars.
    columns (list): Una lista de nombres de columnas para las cuales se generarán los boxplots.
    """
    # Convertir las columnas a pandas para la compatibilidad con matplotlib
    df_pandas = df.select(columns).to_pandas()
    
    # Crear una figura con subplots dispuestos en una fila
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(len(columns) * 4, 6))

    # Si solo hay una columna, axes no es un array, lo convertimos a uno
    if len(columns) == 1:
        axes = [axes]

    # Generar un boxplot para cada columna
    for ax, col in zip(axes, columns):
        ax.boxplot(df_pandas[col].dropna(), vert=False, patch_artist=True)
        ax.set_title(f'Boxplot de {col}')
        ax.set_xlabel(col)

    plt.tight_layout()
    plt.show()
def plot_weekly_sum(df, columns,group_by_period="weeks",xlabel_name="Weeks", ylabel_name="Sum of total values", title_name="Sum of Total Values over Time"):
    # Group by weeks and calculate sum for each column
    weekly_sum = df.group_by(group_by_period).agg(
        [pl.col(col).sum().alias(f"total_{col}") for col in columns]
    )

    # Sort the DataFrame by weeks
    weekly_sum = weekly_sum.sort(group_by_period)

    # Create a line plot
    plt.figure(figsize=(14, 8))
    for col in columns:
        plt.plot(weekly_sum[group_by_period], weekly_sum[f"total_{col}"], label=f"Total {col}")
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.title(title_name)
    plt.legend()
    plt.show()
def create_baskets(
    df: pl.DataFrame,
    group_cols: list[str],
    list_col: str,
    sum_cols: list[str],
    sort_cols: list[str]
) -> pl.DataFrame:
    """
    Process a DataFrame of transactions to create a dataframe of baskets by grouping by specified columns,
    aggregating one column's values into a list, and summing the specified numeric columns,
    and sorting the result by the specified columns.

    Parameters:
    df (pl.DataFrame): The original Polars DataFrame.
    group_cols (list[str]): List of column names to group by.
    list_col (str): The column whose values should be aggregated into a list.
    sum_cols (list[str]): List of numeric column names on which to perform a sum.
    sort_cols (list[str]): List of column names to sort the result by.

    Returns:
    pl.DataFrame: The resulting grouped and aggregated Polars DataFrame.
    """
    agg_exprs = [
        pl.col(list_col).alias(list_col),
        pl.col(list_col).count().alias(f"{list_col}_count")
    ] + [pl.col(col).sum().alias(col) for col in sum_cols]
    
    result = df.group_by(group_cols).agg(agg_exprs)
    
    return result.sort(sort_cols)
def process_baskets_weekly(
    baskets: pl.DataFrame, 
    count_column: str, 
    unique_column: str, 
    mean_columns: list[str]
) -> pl.DataFrame:
    """
    Process the baskets DataFrame to create all possible combinations of weeks and account_id,
    perform aggregation, and join the results.

    Args:
    baskets (pl.DataFrame): Input DataFrame
    count_column (str): Column to count
    unique_column (str): Column to calculate unique values
    mean_columns (list[str]): Columns to calculate mean

    Returns:
    pl.DataFrame: Processed DataFrame
    """
    # Extract the week from the invoice_date column
    baskets = baskets.with_columns([
        pl.col("invoice_date").dt.strftime("%Y-%W").alias("weeks")
    ])
    # Create all possible combinations of weeks and account_id
    all_weeks = baskets.select(pl.col("weeks").unique())
    all_accounts = baskets.select(pl.col("account_id").unique())
    all_combinations = all_weeks.join(all_accounts, how="cross")

    # Perform the aggregation
    aggregations = [
        pl.count(count_column).alias(f"frec_{count_column}"),
        pl.n_unique(unique_column).alias(f"total_{unique_column}"),
    ]
    for column in mean_columns:
        aggregations.append(pl.col(column).mean().alias(f"avg_{column}"))

    result = baskets.group_by(["weeks", "account_id"]).agg(aggregations)

    # Join all possible combinations with the aggregated results
    weekly_result = (
        all_combinations.join(
            result,
            on=["weeks", "account_id"],
            how="left"
        )
        .fill_null(0)
        .sort(["weeks", "account_id"])
    )

    return weekly_result
def process_baskets_biweekly(
    baskets: pl.DataFrame, 
    count_column: str, 
    unique_column: str, 
    mean_columns: list[str]
) -> pl.DataFrame:
    """
    Process the baskets DataFrame to create all possible combinations of biweekly periods and account_id,
    perform aggregation, and join the results.

    Args:
    baskets (pl.DataFrame): Input DataFrame
    count_column (str): Column to count
    unique_column (str): Column to calculate unique values
    mean_columns (list[str]): Columns to calculate mean

    Returns:
    pl.DataFrame: Processed DataFrame
    """
    # Extract the week from the invoice_date column
    baskets = baskets.with_columns([
        pl.col("invoice_date").dt.strftime("%Y-%W").alias("weeks")
    ])

    # Function to assign 2-week periods
    def assign_biweekly(week_str):
        year, week = map(int, week_str.split('-'))
        biweekly = (week - 1) // 2 + 1
        return f"{year}-{biweekly:02d}"

    # Add a new column for biweekly periods
    baskets = baskets.with_columns([
        pl.col("weeks").map_elements(assign_biweekly, return_dtype=pl.Utf8()).alias("biweekly")
    ])

    # Create all possible combinations of biweekly periods and account_id
    all_biweekly = baskets.select(pl.col("biweekly").unique())
    all_accounts = baskets.select(pl.col("account_id").unique())
    all_combinations = all_biweekly.join(all_accounts, how="cross")

    # Perform the aggregation
    aggregations = [
        pl.count(count_column).alias(f"frec_{count_column}"),
        pl.n_unique(unique_column).alias(f"total_{unique_column}"),
    ]
    for column in mean_columns:
        aggregations.append(pl.col(column).mean().alias(f"avg_{column}"))

    result = baskets.group_by(["biweekly", "account_id"]).agg(aggregations)

    # Join all possible combinations with the aggregated results
    biweekly_result = (
        all_combinations.join(
            result,
            on=["biweekly", "account_id"],
            how="left"
        )
        .fill_null(0)
        .sort(["biweekly", "account_id"])
    )

    return biweekly_result
def process_baskets_monthly(
    baskets: pl.DataFrame, 
    count_column: str, 
    unique_column: str, 
    mean_columns: list[str]
) -> pl.DataFrame:
    # Extract the month from the invoice_date column
    baskets = baskets.with_columns([
        pl.col("invoice_date").dt.strftime("%Y-%m").alias("month")
    ])

    # Create all possible combinations of months and account_id
    all_months = baskets.select(pl.col("month").unique())
    all_accounts = baskets.select(pl.col("account_id").unique())
    all_combinations = all_months.join(all_accounts, how="cross")

    # Perform the aggregation
    aggregations = [
        pl.count(count_column).alias(f"frec_{count_column}"),
        pl.n_unique(unique_column).alias(f"total_{unique_column}"),
    ]
    for column in mean_columns:
        aggregations.append(pl.col(column).mean().alias(f"avg_{column}"))

    result = baskets.group_by(["month", "account_id"]).agg(aggregations)

    # Join all possible combinations with the aggregated results
    monthly_result = (
        all_combinations.join(
            result,
            on=["month", "account_id"],
            how="left"
        )
        .fill_null(0)
        .sort(["month", "account_id"])
    )

    return monthly_result
def find_repetitive_purchasers(df, user_id_col, sku_id_col, date_col):
    return df.select([
        user_id_col, 
        sku_id_col, 
        date_col
    ]).unique().group_by([user_id_col, sku_id_col]).agg(
        pl.col("*").count().alias("total_purchases")
    ).filter(
        pl.col("total_purchases") >= 12
    )
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
# This are the functions used in modeling_poc1_TIFUKNN+clustering.ipynb file
def build_item_user_matrix_by_cluster(df: pd.DataFrame, cluster_col: str = 'cluster', sku_col: str = 'sku_id', user_col: str = 'account_id', qty_col: str = 'items_phys_cases') -> Dict[Any, Dict[str, Dict[str, int]]]:
    """
    This function builds a dictionary of item-user matrices, where each key represents a cluster ID and the corresponding value is another dictionary. 
    The inner dictionary maps SKU IDs to user IDs, with the value being the total quantity of items purchased by the user for that SKU in the cluster.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing cluster, SKU, user, and quantity information.
    cluster_col (str): The column name in the DataFrame that represents the cluster ID (default is 'cluster').
    sku_col (str): The column name in the DataFrame that represents the SKU ID (default is 'sku_id').
    user_col (str): The column name in the DataFrame that represents the user ID (default is 'account_id').
    qty_col (str): The column name in the DataFrame that represents the quantity of items purchased (default is 'items_phys_cases').
    
    Returns:
    Dict[Any, Dict[str, Dict[str, int]]]: A dictionary where each key is a cluster ID, and the corresponding value is another dictionary.
    The inner dictionary has SKU IDs as keys and user IDs as values, with the value being the total quantity of items purchased by the user for that SKU in the cluster.
    """

    cluster_item_user_matrices = {}

    # Group by clusters
    for cluster_id in df[cluster_col].unique():
        cluster_df = df[df[cluster_col] == cluster_id]
        item_user_matrix = defaultdict(lambda: defaultdict(int))
        
        for _, row in cluster_df.iterrows():
            item_user_matrix[row[sku_col]][row[user_col]] += row[qty_col]
        
        cluster_item_user_matrices[cluster_id] = item_user_matrix

    return cluster_item_user_matrices
def compute_similarity_by_cluster(cluster_item_user_matrices: Dict[Any, Dict[str, Dict[str, int]]], 
                                  df: pd.DataFrame, 
                                  cluster_col: str = 'cluster', 
                                  user_col: str = 'account_id') -> Dict[Any, Tuple[np.ndarray, list]]:
    """
    This Python function `compute_similarity_by_cluster` calculates the cosine similarity between items in each cluster based on the item-user matrices. 
    The function takes in a dictionary of item-user matrices, a DataFrame containing cluster, SKU, user, and quantity information, and the column names for the cluster and user IDs. 
    It returns a dictionary where each key is a cluster ID, and the corresponding value is a tuple containing the cosine similarity matrix between items in the cluster as a numpy array, 
    and the list of actual SKUs in the cluster. 
    
    - The function iterates over each cluster ID and its corresponding item-user matrix. 
    - It gets the list of users for the current cluster and the list of actual SKUs in the cluster. 
    - It then creates the item-user matrix as a numpy array, and computes the cosine similarity between items using the `cosine_similarity` function. 
    - The resulting cosine similarity matrix is stored in the `cluster_item_similarity` dictionary with the current cluster ID as the key.
    - The function returns the `cluster_item_similarity` dictionary.
    - The cosine similarity matrix is computed using the `cosine_similarity` function, which is not defined in the provided code snippet. 
    - The function returns a dictionary where each key is a cluster ID, and the corresponding value is a tuple containing the cosine similarity matrix 
      between items in the cluster as a numpy array, and the list of actual SKUs in the cluster.
    
    Parameters:
    cluster_item_user_matrices (Dict[Any, Dict[str, Dict[str, int]]]): A dictionary where each key is a cluster ID, and the corresponding value is another dictionary.
    The inner dictionary has SKU IDs as keys and user IDs as values, with the value being the total quantity of items purchased by the user for that SKU in the cluster.
    df (pd.DataFrame): The input DataFrame containing cluster, SKU, user, and quantity information.
    cluster_col (str): The column name in the DataFrame that represents the cluster ID (default is 'cluster').
    user_col (str): The column name in the DataFrame that represents the user ID (default is 'account_id').
    
    Returns:
    Dict[Any, Tuple[np.ndarray, list]]: A dictionary where each key is a cluster ID, and the corresponding value is a tuple.
    The tuple contains the cosine similarity matrix between items in the cluster as a numpy array, and the list of actual SKUs in the cluster.
    """
    cluster_item_similarity = {}

    for cluster_id, item_user_matrix in cluster_item_user_matrices.items():
        # Get the list of users for the current cluster
        users = list(df[df[cluster_col] == cluster_id][user_col].unique())
        # Get the list of actual SKUs in the current cluster
        items = list(item_user_matrix.keys())

        # Create the item-user matrix as a numpy array
        item_user_vectors = np.array([
            [item_user_matrix[item].get(user, 0) for user in users]
            for item in items
        ])

        # Compute the cosine similarity between items
        item_similarity = cosine_similarity(item_user_vectors)
        cluster_item_similarity[cluster_id] = (item_similarity, items)

    return cluster_item_similarity
def predict_next_baskets_for_all(test_data: pd.DataFrame, 
                                 cluster_item_similarity: Dict[Any, Tuple[np.ndarray, list]], 
                                 train_data: pd.DataFrame, 
                                 user_col: str = 'account_id', 
                                 cluster_col: str = 'cluster', 
                                 sku_col: str = 'sku_id', 
                                 k: int = 5):
    """
    This code snippet is a function named `predict_next_baskets_for_all` that predicts the next basket for all users in a test dataset based on item similarity within clusters. 
    The function takes several parameters, including the test dataset, cluster item similarity matrix, training dataset, and column names for user IDs, cluster assignments, and SKUs.
    
    - The function first initializes several dictionaries to store the predicted baskets, ground truth baskets, user histories, and counts for the main strategy and fallback strategies.
    - It then loops through each user in the test dataset, retrieves their cluster, similarity matrix, and item list, and counts the frequency of each item in their purchase history.
    - The function generates recommendations based on item similarity within the cluster and sorts them by score. 
      -If the number of recommendations is less than the desired number `k`, it applies fallback strategies,
        - first try including recommending the most common items purchased by the user that are also common in the cluster 
       -  then,If the number of recommendations is less than the desired number `k`, it applies the most common items purchased only by the user.
    Finally, the function stores the top-k items in the predicted baskets, the ground truth basket for each user, and returns the predicted baskets, 
    ground truth baskets, user histories, and counts for the main strategy and fallback strategies.

    Parameters:
    - test_data (pd.DataFrame): The test dataset containing user information and purchase history.
    - cluster_item_similarity (Dict[Any, Tuple[np.ndarray, list]]): A dictionary mapping each cluster to its item similarity matrix and item list.
    - train_data (pd.DataFrame): The training dataset containing user purchase history.
    - user_col (str): The column name for user IDs in the dataset (default is 'account_id').
    - cluster_col (str): The column name for cluster assignments in the dataset (default is 'cluster').
    - sku_col (str): The column name for SKUs in the dataset (default is 'sku_id').
    - k (int): The number of top items to recommend (default is 5).

    Returns:
    - predicted_baskets (dict): A dictionary mapping each user to their predicted basket.
    - ground_truth_baskets (dict): A dictionary mapping each user to their ground truth basket.
    - user_histories (dict): A dictionary mapping each user to their purchase history.
    - main_strategy_count (int): The number of times the main strategy was used.
    - fallback1_count (int): The number of times the first fallback strategy was used.
    - fallback2_count (int): The number of times the second fallback strategy was used.
    """
    
    predicted_baskets = {}
    ground_truth_baskets = {}
    user_histories = {}
    
    main_strategy_count = 0
    fallback1_count = 0
    fallback2_count = 0

    # Loop through each user in the test dataset
    for account_id in test_data[user_col].unique():
        # Get the cluster of the user
        user_cluster = test_data[test_data[user_col] == account_id][cluster_col].iloc[0]
        
        # Get the similarity matrix and item list for the user's cluster
        item_similarity, items = cluster_item_similarity[user_cluster]
        
        # Retrieve the user's purchase history from the training data
        user_history = train_data[train_data[user_col] == account_id][sku_col].tolist()
        user_histories[account_id] = user_history
        
        if len(user_history) == 0:
            continue
        
        # Count the frequency of each item in the user's history
        item_freq = defaultdict(int)
        for item in user_history:
            if item in items:
                item_freq[item] += 1

        # Generate recommendations based on item similarity within the cluster
        recommendations = []
        for item, freq in item_freq.items():
            if item in items:
                similar_items_idx = np.argsort(item_similarity[items.index(item)])[::-1][:k]
                recommendations.extend([(items[i], item_similarity[items.index(item), i] * freq) for i in similar_items_idx])

        # Sort recommendations by score
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        final_recommendations = [r[0] for r in recommendations[:k]]

        # If the number of recommendations is less than k, apply fallback strategies
        if len(final_recommendations) < k:
            # Fallback 1: Most common items purchased by the user that are also common in the cluster
            cluster_common_items = Counter([item for user in train_data[train_data[cluster_col] == user_cluster][user_col].unique() 
                                            for item in train_data[train_data[user_col] == user][sku_col].tolist()])
            user_common_items = Counter(user_history)
            common_items = [item for item in user_common_items if item in cluster_common_items]
            common_items = sorted(common_items, key=lambda x: cluster_common_items[x], reverse=True)
            final_recommendations.extend(common_items[:k - len(final_recommendations)])
            
            if len(final_recommendations) < k:
                # Fallback 2: Most common items purchased only by the user
                user_specific_items = [item for item in user_common_items if item not in common_items]
                final_recommendations.extend(user_specific_items[:(k - len(final_recommendations))])
                fallback2_count += 1
            else:
                fallback1_count += 1
        else:
            main_strategy_count += 1

        # Store the top-k items in predicted_baskets
        predicted_baskets[account_id] = final_recommendations[:k]
        
        # Store the ground truth basket for the user
        ground_truth_baskets[account_id] = test_data[test_data[user_col] == account_id][sku_col].tolist()

    return predicted_baskets, ground_truth_baskets, user_histories, main_strategy_count, fallback1_count, fallback2_count
def predict_next_basket_for_user(account_id: str, 
                                 test_data: pd.DataFrame, 
                                 cluster_item_similarity: Dict[Any, Tuple[np.ndarray, list]], 
                                 train_data: pd.DataFrame, 
                                 user_col: str = 'account_id', 
                                 cluster_col: str = 'cluster', 
                                 sku_col: str = 'sku_id', 
                                 k: int = 5):
    """
    
    This code snippet is a function named `predict_next_basket_for_user` that predicts the next basket for a specific user based on item similarity within their cluster. 
    It takes into account the user's purchase history and applies fallback strategies if the primary prediction method does not fill the required number of recommendations. 
    The function returns a list of the top-k recommended SKUs for the user.

    Parameters:
    - account_id (str): The ID of the user to generate the prediction for.
    - test_data (pd.DataFrame): The test dataset containing information about users and their transactions.
    - cluster_item_similarity (dict): A dictionary containing item similarity matrices for each cluster.
    - train_data (pd.DataFrame): The training dataset containing historical transactions.
    - user_col (str): The column name for user IDs in the dataset.
    - cluster_col (str): The column name for cluster IDs in the dataset.
    - sku_col (str): The column name for SKU/item IDs in the dataset.
    - k (int): The number of recommendations to generate.

    Returns:
    - list: A list of the top-k recommended SKUs for the user.
    """
    # Check if the user exists in the test dataset
    if account_id not in test_data[user_col].values:
        raise ValueError(f"Account ID {account_id} does not exist in the test dataset.")
    
    # Get the cluster of the user
    user_cluster = test_data[test_data[user_col] == account_id][cluster_col].iloc[0]
    
    # Get the similarity matrix and item list for the user's cluster
    item_similarity, items = cluster_item_similarity[user_cluster]
    
    # Retrieve the user's purchase history from the training data
    user_history = train_data[train_data[user_col] == account_id][sku_col].tolist()
    
    if len(user_history) == 0:
        raise ValueError(f"No purchase history found for Account ID {account_id}.")
    
    # Count the frequency of each item in the user's history
    item_freq = defaultdict(int)
    for item in user_history:
        if item in items:
            item_freq[item] += 1

    # Generate recommendations based on item similarity within the cluster
    recommendations = []
    for item, freq in item_freq.items():
        if item in items:
            similar_items_idx = np.argsort(item_similarity[items.index(item)])[::-1][:k]
            recommendations.extend([(items[i], item_similarity[items.index(item), i] * freq) for i in similar_items_idx])

    # Sort recommendations by score
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    final_recommendations = [r[0] for r in recommendations[:k]]

    # If the number of recommendations is less than k, apply fallback strategies
    if len(final_recommendations) < k:
        # Fallback 1: Most common items purchased by the user that are also common in the cluster
        cluster_common_items = Counter([item for user in train_data[train_data[cluster_col] == user_cluster][user_col].unique() 
                                        for item in train_data[train_data[user_col] == user][sku_col].tolist()])
        user_common_items = Counter(user_history)
        common_items = [item for item in user_common_items if item in cluster_common_items]
        common_items = sorted(common_items, key=lambda x: cluster_common_items[x], reverse=True)
        final_recommendations.extend(common_items[:k - len(final_recommendations)])
        
        if len(final_recommendations) < k:
            # Fallback 2: Most common items purchased only by the user
            user_specific_items = [item for item in user_common_items if item not in common_items]
            final_recommendations.extend(user_specific_items[:(k - len(final_recommendations))])

    return final_recommendations[:k]
# This are the functions used in nmodeling_poc2_TIFUKNN.ipynb file
def split_train_test_iso(transactions):
    """
    This function splits a transactions DataFrame into training and testing sets based on biweekly periods. 
    It assigns the most recent biweekly period to the test set and the rest to the training set.

    Parameters:
        transactions (pandas.DataFrame): The DataFrame containing the transactions data.

    Returns:
        tuple: A tuple containing two DataFrames. The first DataFrame is the training set, and the second DataFrame is the testing set.
    """
    transactions['biweekly_period'] = transactions['invoice_date'].dt.isocalendar().week // 2
    max_period = transactions['biweekly_period'].max()
    train_data = transactions[transactions['biweekly_period'] < max_period]
    test_data = transactions[transactions['biweekly_period'] == max_period]

    return train_data, test_data
def exponential_decay(max_period, current_period, group_decay_rate, within_group_decay_rate, is_same_group, alpha):
    """
    This function calculates a decay weight for a transaction based on the time difference between the maximum period and the current period. 
    The decay rate depends on whether the transaction is within the same group or across different groups. 
    The general decay factor `alpha` is also applied.

    Args:
    - max_period (int): The maximum period in the dataset.
    - current_period (int): The current period for the transaction.
    - group_decay_rate (float): The decay rate for interactions across different groups.
    - within_group_decay_rate (float): The decay rate for interactions within the same group.
    - is_same_group (bool): Whether the interaction is within the same group or across different groups.
    - alpha (float): The general decay factor.

    Returns:
    - float: The decay weight for the current transaction.
    """
    time_diff = max_period - current_period
    if is_same_group:
        return alpha * (within_group_decay_rate ** time_diff)
    else:
        return alpha * (group_decay_rate ** time_diff)
    
    return alpha ** time_diff
def compute_item_similarity_iso(train_data, decay_function, user_col, item_col, period_col, quantity_col, 
                                alpha, group_decay_rate=1, within_group_decay_rate=1, group_col=None):
    """
   This function calculates the similarity between items (SKUs) based on user purchase histories. 
   It takes into account the time decay of interactions, where more recent interactions have a greater impact on the similarity calculation. 
   The function returns a similarity matrix and a list of items (SKUs).

    Args:
    - train_data (DataFrame): The training data containing user purchase histories.
    - decay_function (function): A function that applies the time decay based on the period difference.
    - user_col (str): The column name for user IDs.
    - item_col (str): The column name for item (SKU) IDs.
    - period_col (str): The column name for the biweekly periods.
    - quantity_col (str): The column name for the quantity of items purchased.
    - alpha (float): The general time decay parameter.
    - group_col (str): The column name for grouping items (optional).
    - group_decay_rate (float): The decay rate for interactions across different groups. (needed when group_col is provided)
    - within_group_decay_rate (float): The decay rate for interactions within the same group.(needed when group_col is not provided)

    Returns:
    - item_similarity (np.ndarray): The item similarity matrix.
    - items (list): The list of items (SKUs).
    """
    item_user_matrix = defaultdict(lambda: defaultdict(float))
    users = train_data[user_col].unique()

    max_period = train_data[period_col].max()

    for _, row in train_data.iterrows():
        if group_col:
            # Determine if the interaction is within the same group or not
            is_same_group = (train_data[group_col].max() == row[group_col])
        else:
            is_same_group = True  # Default to within the same group if no group column is provided
        
        time_decay = decay_function(max_period, row[period_col], group_decay_rate, within_group_decay_rate, is_same_group,alpha)
        item_user_matrix[row[item_col]][row[user_col]] += row[quantity_col] * time_decay

    items = list(item_user_matrix.keys())
    item_vectors = np.array([
        [item_user_matrix[item].get(user, 0) for user in users]
        for item in items
    ])
    item_similarity = cosine_similarity(item_vectors)

    return item_similarity, items
def select_k(item_similarity, random_state=11):
    """
    This Python function, `select_k`, is used to determine the optimal number of clusters (k) for an item similarity matrix. 

    The function works as follows:
    
    0. The function is designed to work with an item similarity matrix.
    1. It iterates over a range of k values from 2 to 10.
    2. For each k, it performs KMeans clustering on the item similarity matrix.
    3. It calculates the Sum of Squared Errors (SSE) for each k.
    4. It calculates the differences between consecutive SSE values.
    5. It calculates the differences between consecutive changes in SSE.
    6. The optimal k is determined by finding the k corresponding to the minimum change between consecutive changes.
    7. The function returns the optimal k and SSE values.

    Parameters:
    item_similarity (numpy.ndarray): The item similarity matrix.

    Returns:
    int: The optimal number of clusters (k).
    list: SSE values for each k.
    """
    sse = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(item_similarity)
        sse.append(kmeans.inertia_)
    
    # Calculate the differences between consecutive SSE values
    sse_diff = np.diff(sse)
    
    # Calculate the differences between consecutive changes in SSE
    sse_diff_changes = np.diff(sse_diff)
    
    # Find the k corresponding to the minimum change between consecutive changes
    optimal_k = np.argmin(sse_diff_changes) + 2  # +2 because k starts from 2 and we take the change of the change
    
    return optimal_k, sse,sse_diff_changes
def find_most_similar_items(item_similarity, items, k, random_state=11):
    """
    This code snippet uses K-Means clustering to group similar items together based on their similarity matrix. 
    It takes in an item similarity matrix, a list of items, and the number of clusters (k) as input, 
    and returns a dictionary mapping each item to its corresponding cluster.

    Parameters:
    item_similarity (numpy.ndarray): The item similarity matrix.
    items (list): The list of items.
    k (int): The number of clusters.

    Returns:
    dict: A dictionary mapping each item to its cluster.
    """
    kmeans = KMeans(n_clusters=k, random_state=45)
    kmeans.fit(item_similarity)
    clusters = kmeans.predict(item_similarity)

    item_to_cluster = {item: cluster for item, cluster in zip(items, clusters)}
    return item_to_cluster
def generate_recommendations(user_history, item_similarity, items, item_to_cluster, k,threshold):
    """
    This function generates personalized item recommendations based on a user's interaction history and item similarity. It works as follows:

    1. For each item in the user's history, it finds the cluster it belongs to.
    2. It then identifies similar items within that cluster that have a similarity score above a given threshold.
    3. The function assigns a score to each similar item based on its frequency of appearance in the user's history.
    4. Finally, it returns the top k items with the highest scores as recommendations.

    In essence, this function uses collaborative filtering and item similarity to provide personalized recommendations.

    Parameters:
    user_history (list): A list of items the user has previously interacted with.
    item_similarity (numpy.ndarray): The item similarity matrix.
    items (list): The list of items.
    item_to_cluster (dict): A dictionary mapping each item to its cluster.
    k (int): The number of recommendations to return.
    threshold (int): The threshold for the similarity of items

    Returns:
    list: A list of the top k recommended items based on the user's history and item similarity.
    """
    item_scores = Counter()
    for item in user_history:
        if item in item_to_cluster:
            cluster = item_to_cluster[item]
            similar_items = [items[i] for i in np.where(item_similarity[cluster] > threshold)[0]]
            for sim_item in similar_items:
                item_scores[sim_item] += 1

    return [item for item, score in item_scores.most_common(k)]
def generate_basket_data(train_data, test_data, item_similarity, items, item_to_cluster, user_col='account_id', item_col='sku_id', k=5,threshold=0.5):
    """
    This function generates predicted baskets, ground truth baskets, and user histories for all users in the test set. 
    It takes in training data, test data, item similarity matrix, list of items, item-to-cluster mapping, 
    and parameters for the recommendation algorithm (k and threshold). 
    The function iterates over each user in the test set, retrieves their purchase history from the training data, 
    their ground truth basket from the test data, and generates a predicted basket using the `generate_recommendations` function. 
    The function returns three dictionaries: one for predicted baskets, one for ground truth baskets, and one for user histories.

    Args:
    - train_data (DataFrame): The training data containing user purchase histories.
    - test_data (DataFrame): The test data containing the ground truth baskets.
    - item_similarity (np.ndarray): The item similarity matrix.
    - items (list): The list of all items.
    - item_to_cluster (dict): Mapping of items to their clusters.
    - user_col (str): The name of the user column in the dataset.
    - item_col (str): The name of the item (SKU) column in the dataset.
    - k (int): The number of top items to recommend.
    - threshold (int): The threshold for the similarity of items

    Returns:
    - predicted_baskets (dict): Dictionary with users as keys and their predicted baskets as values.
    - ground_truth_baskets (dict): Dictionary with users as keys and their ground truth baskets as values.
    - user_histories (dict): Dictionary with users as keys and their purchase histories as values.
    """
    predicted_baskets = {}
    ground_truth_baskets = {}
    user_histories = {}

    for account_id in test_data[user_col].unique():
        user_history = train_data[train_data[user_col] == account_id][item_col].tolist()
        ground_truth = test_data[test_data[user_col] == account_id][item_col].tolist()
        predicted_basket = generate_recommendations(user_history, item_similarity, items, item_to_cluster, k,threshold)

        predicted_baskets[account_id] = predicted_basket
        ground_truth_baskets[account_id] = ground_truth
        user_histories[account_id] = user_history

    return predicted_baskets, ground_truth_baskets, user_histories
# Mixed strategy functions
def predict_nb_for_all_mixed_strategy(
    test_data: pd.DataFrame,
    cluster_item_similarity: Dict[Any, Tuple[np.ndarray, list]],
    train_data: pd.DataFrame,
    tifuknn_strategy: Callable[[Dict[str, int], np.ndarray, list, int], list],
    strategy_used: Callable[[list, list, list], str],
    user_col: str = 'account_id',
    cluster_col: str = 'cluster',
    sku_col: str = 'sku_id',
    k: int = 5,
    strategy_1_count: int = 2,
    strategy_2_count: int = 2
) -> Tuple[Dict[str, list], Dict[str, list], Dict[str, list], Dict[str, int]]:
    """
    This code snippet is a function named `predict_nb_for_all_mixed_strategy` that predicts the next basket for all users in a test dataset based on item similarity within clusters. 
    The basket is assembled using three strategies in a controlled order:

    1. Strategy 1: Most frequent items that are common between the user and the cluster.
    2. Strategy 2: Most frequent items purchased only by the user (but not included in Strategy 1).
    3. Strategy 3: Fill the remaining slots using the provided TIFUKNN strategy.

    The function takes several parameters, including the test dataset, cluster item similarity matrix, training dataset, 
    TIFUKNN strategy, and strategy used. It returns four dictionaries: predicted baskets, ground truth baskets, user histories, and strategy counts.

    Parameters:
    - test_data (pd.DataFrame): The test dataset containing user information and purchase history.
    - cluster_item_similarity (Dict[Any, Tuple[np.ndarray, list]]): A dictionary mapping each cluster to its item similarity matrix and item list.
    - train_data (pd.DataFrame): The training dataset containing user purchase history.
    - tifuknn_strategy (Callable): A function to compute recommendations using the TIFUKNN algorithm.
    - strategy_used (Callable): A function to determine which strategy was used for each user.
    - user_col (str): The column name for user IDs in the dataset (default is 'account_id').
    - cluster_col (str): The column name for cluster assignments in the dataset (default is 'cluster').
    - sku_col (str): The column name for SKUs in the dataset (default is 'sku_id').
    - k (int): The number of top items to recommend (default is 14).
    - strategy_1_count (int): The number of items to include from Strategy 1.
    - strategy_2_count (int): The number of items to include from Strategy 2.

    Returns:
    - predicted_baskets (dict): A dictionary mapping each user to their predicted basket.
    - ground_truth_baskets (dict): A dictionary mapping each user to their ground truth basket.
    - user_histories (dict): A dictionary mapping each user to their purchase history.
    - strategy_counts (dict): A dictionary containing counts of how many users were satisfied by each strategy.
    """
    
    # Initialize storage for results
    predicted_baskets = {}
    ground_truth_baskets = {}
    user_histories = {}
    
    strategy_counts = {'main_strategy_count': 0, 'fallback1_count': 0, 'fallback2_count': 0}
    #count = 0

    # Loop through each user in the test dataset
    for account_id in test_data[user_col].unique():
        # Get the cluster of the user
        user_cluster = test_data[test_data[user_col] == account_id][cluster_col].iloc[0]
        
        # Get the similarity matrix and item list for the user's cluster
        item_similarity, items = cluster_item_similarity[user_cluster]
        
        # Retrieve the user's purchase history from the training data
        user_history = train_data[train_data[user_col] == account_id][sku_col].tolist()
        user_histories[account_id] = user_history
        
        if len(user_history) == 0:
            continue
        
        # Count the frequency of each item in the user's history
        item_freq = defaultdict(int)
        for item in user_history:
            if item in items:
                item_freq[item] += 1

        # Strategy 1: Most frequent items that are common between user and cluster
        cluster_common_items = Counter([item for user in train_data[train_data[cluster_col] == user_cluster][user_col].unique() 
                                        for item in train_data[train_data[user_col] == user][sku_col].tolist()])
        user_common_items = Counter(user_history)
        strategy_1_items = [item for item in user_common_items if item in cluster_common_items]
        strategy_1_items = sorted(strategy_1_items, key=lambda x: cluster_common_items[x], reverse=True)[:strategy_1_count]
        
        # Strategy 2: Most frequent items purchased only by the user (but not included in Strategy 1)
        strategy_2_items = [item for item in user_common_items if item not in strategy_1_items]
        strategy_2_items = sorted(strategy_2_items, key=lambda x: user_common_items[x], reverse=True)[:strategy_2_count]
        
        # Strategy 3: Fill the remaining slots using the provided TIFUKNN strategy
        strategy_3_items = tifuknn_strategy(item_freq, item_similarity, items, k - strategy_1_count - strategy_2_count)
        
        # Assemble the final recommendation list
        final_recommendations = strategy_1_items + strategy_2_items + strategy_3_items[:k - len(strategy_1_items) - len(strategy_2_items)]
        
        # Determine which strategy was used
        strategy_label = strategy_used(strategy_1_items, strategy_2_items, strategy_3_items)
        strategy_counts[strategy_label] += 1

        # Store the top-k items in predicted_baskets
        predicted_baskets[account_id] = final_recommendations[:k]
        
        # Store the ground truth basket for the user
        ground_truth_baskets[account_id] = test_data[test_data[user_col] == account_id][sku_col].tolist()
        
        #count+=1
        #print(count)

    return predicted_baskets, ground_truth_baskets, user_histories, strategy_counts
def tifuknn_strategy(user_history, item_similarity, items, remaining_k):
    """
    This Python function implements the TIFUKNN strategy to generate item recommendations based on the user's history and item similarity. 
    The function takes in the user's history, the item similarity matrix, and the number of top items to recommend and do this process:
    
    - The function first counts the frequency of each item in the user's history. 
    - It then generates recommendations by iterating through the user's history and item similarity matrix. 
    - For each item, it finds the top recommended items based on the user's history and item similarity. 
    - The function then returns a list of the top recommended items based on the user's history and item similarity

    Parameters:
    user_history (list): A list of items the user has interacted with.
    item_similarity (numpy.ndarray): The item similarity matrix.
    items (list): The list of all items.
    remaining_k (int): The number of top items to recommend.

    Returns:
    list: A list of the top recommended items based on the user's history and item similarity.
    """
    item_freq = Counter(user_history)
    recommendations = []
    for item, freq in item_freq.items():
        if item in items:
            similar_items_idx = np.argsort(item_similarity[items.index(item)])[::-1][:remaining_k]
            recommendations.extend([(items[i], item_similarity[items.index(item), i] * freq) for i in similar_items_idx])
    return [r[0] for r in sorted(recommendations, key=lambda x: x[1], reverse=True)]

def strategy_used(strategy_1, strategy_2, strategy_3):
    """
    This code defines a function called `strategy_used` that takes in three lists: `strategy_1`, `strategy_2`, and `strategy_3`. 
    The function determines which strategy was used for item recommendation based on the lengths of the three lists. 
    
    - If `strategy_3` has a length greater than 0, the function returns the string `'main_strategy_count'`. 
    - If `strategy_2` has a length greater than 0, the function returns the string `'fallback1_count'`. 
    - Otherwise, the function returns the string `'fallback2_count'`.

    Parameters:
    strategy_1 (list): The list of items recommended by the primary strategy.
    strategy_2 (list): The list of items recommended by the secondary strategy.
    strategy_3 (list): The list of items recommended by the tertiary strategy.

    Returns:
    str: The label of the strategy used, either 'main_strategy_count', 'fallback1_count', or 'fallback2_count'.
    """
    if len(strategy_3) > 0:
        return 'main_strategy_count'
    elif len(strategy_2) > 0:
        return 'fallback1_count'
    else:
        return 'fallback2_count'

# This are the metrics used to evaluate our experiments
# Conventional Metrics
def precision_at_k(predicted_basket, ground_truth_basket, k):
    """
    Calculate Precision@K for the given predicted basket and ground truth basket.
    Precision@K measures the fraction of relevant items among the top K recommendations.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - ground_truth_basket (list): List of items actually purchased/consumed by the user.
    - k (int): The number of top items to consider in the predicted basket.

    Returns:
    - float: Precision@K score.
    """
    predicted_items = set(predicted_basket[:k])
    relevant_items = set(ground_truth_basket)
    true_positives = len(predicted_items & relevant_items)
    return true_positives / k if k > 0 else 0.0

def f1_at_k(predicted_basket, ground_truth_basket, k,precision_at_k,recall_at_k):
    """
    Calculate F1@K for the given predicted basket and ground truth basket.

    F1@K is the harmonic mean of Precision@K and Recall@K.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - ground_truth_basket (list): List of items actually purchased/consumed by the user.
    - k (int): The number of top items to consider in the predicted basket.

    Returns:
    - float: F1@K score.
    """
    precision = precision_at_k(predicted_basket, ground_truth_basket, k)
    recall = recall_at_k(predicted_basket, ground_truth_basket, k)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def recall_at_k(predicted_basket, ground_truth_basket, k):
    """
    Calculate Recall@K for the given predicted basket and ground truth basket.

    Recall@K measures the fraction of relevant items in the ground truth that are present in the top K recommendations.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - ground_truth_basket (list): List of items actually purchased/consumed by the user.
    - k (int): The number of top items to consider in the predicted basket.

    Returns:
    - float: Recall@K score.
    """
    relevant_items = set(ground_truth_basket)
    predicted_items = set(predicted_basket[:k])
    return len(relevant_items & predicted_items) / len(relevant_items) if relevant_items else 0.0

def ndcg_at_k(predicted_basket, ground_truth_basket, k):
    """
    Calculate NDCG@K for the given predicted basket and ground truth basket.

    NDCG@K (Normalized Discounted Cumulative Gain) measures the ranking quality of the top K recommendations,
    taking into account the order of items.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - ground_truth_basket (list): List of items actually purchased/consumed by the user.
    - k (int): The number of top items to consider in the predicted basket.

    Returns:
    - float: NDCG@K score.
    """
    dcg = sum([1 / np.log2(i + 2) if predicted_basket[i] in ground_truth_basket else 0 for i in range(min(k, len(predicted_basket)))])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(ground_truth_basket)))])
    return dcg / idcg if idcg > 0 else 0.0

def phr_at_k(predicted_basket, ground_truth_basket, k):
    """
    Calculate PHR@K (Personalized Hit Ratio) for the given predicted basket and ground truth basket.

    PHR@K measures the proportion of users for whom at least one item in the top K recommendations is in the ground truth.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - ground_truth_basket (list): List of items actually purchased/consumed by the user.
    - k (int): The number of top items to consider in the predicted basket.

    Returns:
    - float: PHR@K score (1.0 if at least one item matches, otherwise 0.0).
    """
    return 1.0 if set(predicted_basket[:k]) & set(ground_truth_basket) else 0.0


# Novel Metrics for Repetition and Exploration
def repetition_ratio(predicted_basket, user_history):
    """
    Calculate the Repetition Ratio (RepR) for the given predicted basket and user history.

    RepR measures the proportion of items in the recommended basket that have appeared in the user's history.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - user_history (list): List of items the user has previously interacted with.

    Returns:
    - float: Repetition Ratio (RepR) score.
    """
    repeated_items = [item for item in predicted_basket if item in user_history]
    return len(repeated_items) / len(predicted_basket) if predicted_basket else 0.0

def exploration_ratio(predicted_basket, user_history):
    """
    Calculate the Exploration Ratio (ExplR) for the given predicted basket and user history.

    ExplR measures the proportion of items in the recommended basket that are new to the user.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - user_history (list): List of items the user has previously interacted with.

    Returns:
    - float: Exploration Ratio (ExplR) score.
    """
    new_items = [item for item in predicted_basket if item not in user_history]
    return len(new_items) / len(predicted_basket) if predicted_basket else 0.0

def recall_rep(predicted_basket, ground_truth_basket, user_history):
    """
    Calculate Recallrep for the given predicted basket, ground truth basket, and user history.

    Recallrep measures the recall for repeat items, which are items that the user has interacted with before.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - ground_truth_basket (list): List of items actually purchased/consumed by the user.
    - user_history (list): List of items the user has previously interacted with.

    Returns:
    - float: Recallrep score.
    """
    repeated_ground_truth = [item for item in ground_truth_basket if item in user_history]
    repeated_predictions = [item for item in predicted_basket if item in repeated_ground_truth]
    return len(repeated_predictions) / len(repeated_ground_truth) if repeated_ground_truth else 0.0

def recall_expl(predicted_basket, ground_truth_basket, user_history):
    """
    Calculate Recallexpl for the given predicted basket, ground truth basket, and user history.

    Recallexpl measures the recall for explore items, which are items that are new to the user.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - ground_truth_basket (list): List of items actually purchased/consumed by the user.
    - user_history (list): List of items the user has previously interacted with.

    Returns:
    - float: Recallexpl score.
    """
    new_ground_truth = [item for item in ground_truth_basket if item not in user_history]
    new_predictions = [item for item in predicted_basket if item in new_ground_truth]
    return len(new_predictions) / len(new_ground_truth) if new_ground_truth else 0.0

def phr_rep(predicted_basket, ground_truth_basket, user_history):
    """
    Calculate PHRrep (Personalized Hit Ratio for Repeat items) for the given predicted basket, ground truth basket, and user history.

    PHRrep measures the personalized hit ratio for repeat items, which are items that the user has interacted with before.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - ground_truth_basket (list): List of items actually purchased/consumed by the user.
    - user_history (list): List of items the user has previously interacted with.

    Returns:
    - float: PHRrep score (1.0 if at least one repeat item matches, otherwise 0.0).
    """
    repeated_ground_truth = [item for item in ground_truth_basket if item in user_history]
    return 1.0 if set(predicted_basket) & set(repeated_ground_truth) else 0.0

def phr_expl(predicted_basket, ground_truth_basket, user_history):
    """
    Calculate PHRexpl (Personalized Hit Ratio for Explore items) for the given predicted basket, ground truth basket, and user history.

    PHRexpl measures the personalized hit ratio for explore items, which are items that are new to the user.

    Args:
    - predicted_basket (list): List of items predicted by the recommendation model.
    - ground_truth_basket (list): List of items actually purchased/consumed by the user.
    - user_history (list): List of items the user has previously interacted with.

    Returns:
    - float: PHRexpl score (1.0 if at least one explore item matches, otherwise 0.0).
    """
    new_ground_truth = [item for item in ground_truth_basket if item not in user_history]
    return 1.0 if set(predicted_basket) & set(new_ground_truth) else 0.0

# metric created by me
def precision_at_k_over_n(predicted: list, actual: list, n: int, k: int) -> float:
    """
    Calculates a modified precision at k, where the number of correctly predicted items
    is divided by a given value n instead of the total number of predicted items.

    This function can be used to evaluate precision in scenarios where the standard
    precision formula needs to be adjusted by a factor of n.

    Args:
        predicted (list): A list of items (e.g., SKUs) that are predicted to be in the next basket.
        actual (list): A list of items that are actually in the next basket.
        n (int): The divisor used in the precision calculation. 
        k (int): The number of top predictions to consider.

    Returns:
        float: The modified precision at k, calculated as the number of correctly predicted items
               divided by n. The value ranges depending on the choice of n and the overlap between
               predicted and actual items.

    Example:
        predicted = ['item1', 'item2', 'item3', 'item4']
        actual = ['item2', 'item4', 'item6']
        n = 3
        k = 3
        precision_at_k_over_n(predicted, actual, n, k)
        # Output: 0.3333  (1 correct prediction divided by n=3)

    Notes:
        - The function assumes that both `predicted` and `actual` are lists of items, and that `k` and `n` are positive integers.
        - If k is larger than the length of the predicted list, the function will consider all predicted items.
        - This function is a variation of precision that introduces an external divisor n to control the calculation.
    """
    # Create sets for the predicted and actual items (limited to the top-k predicted items)
    predicted_set = set(predicted[:k])
    actual_set = set(actual)
    
    # Calculate the modified precision at k
    result = len(predicted_set & actual_set) / n
    
    return result

def evaluate_model_metrics(predicted_baskets, ground_truth_baskets, user_histories, k=5, n=3):
    """
    Evaluates various performance metrics for the model's predictions.

    Args:
    - predicted_baskets (dict): A dictionary where keys are user IDs and values are lists of predicted items.
    - ground_truth_baskets (dict): A dictionary where keys are user IDs and values are lists of actual items.
    - user_histories (dict): A dictionary where keys are user IDs and values are lists of past items the user interacted with.
    - k (int): The number of top items to consider in the predictions.
    - n (int): The divisor used in the modified precision at k calculation.

    Returns:
    - dict: A dictionary containing the average scores for each metric.
    """
    # Initialize lists to store metric results
    precision_results = []
    recall_results = []
    f1_results = []
    ndcg_results = []
    phr_results = []
    repetition_results = []
    exploration_results = []
    recall_rep_results = []
    recall_expl_results = []
    phr_rep_results = []
    phr_expl_results = []
    precision_at_k_over_n_results = []

    # Loop through each user and calculate metrics
    for user, predicted in predicted_baskets.items():
        actual = ground_truth_baskets.get(user, [])
        history = user_histories.get(user, [])

        precision = precision_at_k(predicted, actual, k)
        recall = recall_at_k(predicted, actual, k)
        f1 = f1_at_k(predicted, actual, k, precision_at_k, recall_at_k)
        ndcg = ndcg_at_k(predicted, actual, k)
        phr = phr_at_k(predicted, actual, k)
        repetition = repetition_ratio(predicted, history)
        exploration = exploration_ratio(predicted, history)
        recall_rep_score = recall_rep(predicted, actual, history)
        recall_expl_score = recall_expl(predicted, actual, history)
        phr_rep_score = phr_rep(predicted, actual, history)
        phr_expl_score = phr_expl(predicted, actual, history)
        precision_over_n_score = precision_at_k_over_n(predicted, actual, n=n, k=k)

        # Store results
        precision_results.append(precision)
        recall_results.append(recall)
        f1_results.append(f1)
        ndcg_results.append(ndcg)
        phr_results.append(phr)
        repetition_results.append(repetition)
        exploration_results.append(exploration)
        recall_rep_results.append(recall_rep_score)
        recall_expl_results.append(recall_expl_score)
        phr_rep_results.append(phr_rep_score)
        phr_expl_results.append(phr_expl_score)
        precision_at_k_over_n_results.append(precision_over_n_score)

    # Calculate average metrics across all users
    results = {
        'Avg Precision@k': np.mean(precision_results),
        'Avg Recall@k': np.mean(recall_results),
        'Avg F1@k': np.mean(f1_results),
        'Avg NDCG@k': np.mean(ndcg_results),
        'PHR@k': np.mean(phr_results),
        'Avg Repetition Ratio': np.mean(repetition_results),
        'Avg Exploration Ratio': np.mean(exploration_results),
        'Avg Recallrep': np.mean(recall_rep_results),
        'Avg Recallexpl': np.mean(recall_expl_results),
        'PHRrep': np.mean(phr_rep_results),
        'PHRexpl': np.mean(phr_expl_results),
        'Avg Precision@k over n': np.mean(precision_at_k_over_n_results)
    }

    return results

def calculate_basket_statistics(ground_truth_baskets):
    """
    Calculates the average size, 25th, 50th (median), and 75th percentiles of the baskets in the ground truth data.
    Also returns the list of basket sizes.

    Args:
    - ground_truth_baskets (dict): A dictionary where keys are user IDs and values are lists of actual items.

    Returns:
    - dict: A dictionary containing the average size, specified percentiles, and the list of basket sizes.
    """
    basket_sizes = [len(basket) for basket in ground_truth_baskets.values()]

    if len(basket_sizes) == 0:
        return {
            'average_size': 0.0,
            'percentile_25': 0.0,
            'percentile_50': 0.0,
            'percentile_75': 0.0,
            "max_size": 0.0,
            'basket_sizes': basket_sizes
        }

    average_basket_size = np.mean(basket_sizes)
    percentile_25 = np.percentile(basket_sizes, 25)
    percentile_50 = np.percentile(basket_sizes, 50)
    percentile_75 = np.percentile(basket_sizes, 75)
    max_size = np.max(basket_sizes)

    return {
        'average_size': average_basket_size,
        'percentile_25': percentile_25,
        'percentile_50': percentile_50,
        'percentile_75': percentile_75,
        "max_size": max_size,
        'basket_sizes': basket_sizes
    }