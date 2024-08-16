import os
from functools import reduce
from collections import defaultdict, Counter
import polars as pl
import pandas as pd
import pyarrow
pl.Config(tbl_rows=50)
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import kruskal
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

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

def build_item_user_matrix_by_cluster(df: pd.DataFrame, cluster_col: str = 'cluster', sku_col: str = 'sku_id', user_col: str = 'account_id', qty_col: str = 'items_phys_cases') -> Dict[Any, Dict[str, Dict[str, int]]]:
    """
    Builds an item-user matrix for each cluster, where the rows represent SKUs (items)
    and the columns represent users. The values in the matrix are the total quantities of
    items purchased by each user within each cluster.

    Args:
        df (pd.DataFrame): The DataFrame containing transaction data along with cluster assignments.
        cluster_col (str): The name of the column representing the cluster assignment.
        sku_col (str): The name of the column representing the SKUs (items).
        user_col (str): The name of the column representing the user IDs.
        qty_col (str): The name of the column representing the quantity of items purchased.

    Returns:
        Dict: A dictionary where each key is a cluster ID, and the value is another dictionary
              representing the item-user matrix for that cluster. In the item-user matrix:
              - The keys are SKUs (items).
              - The inner keys are user IDs.
              - The inner values are the total quantities of each SKU purchased by the corresponding user.
    
    Example:
        cluster_item_user_matrices = build_item_user_matrix_by_cluster(pd_transactions_cluster, 
                                                                       cluster_col='cluster',
                                                                       sku_col='sku_id',
                                                                       user_col='account_id',
                                                                       qty_col='items_phys_cases')
    """
    cluster_item_user_matrices = {}

    # Group by clusters
    for cluster_id in df[cluster_col].unique():
        cluster_df = df[df[cluster_col] == cluster_id]
        item_user_matrix = defaultdict(lambda: defaultdict(int))
        
        for _, row in cluster_df.iterrows():
            for sku in row[sku_col]:
                item_user_matrix[sku][row[user_col]] += row[qty_col]
        
        cluster_item_user_matrices[cluster_id] = item_user_matrix

    return cluster_item_user_matrices
