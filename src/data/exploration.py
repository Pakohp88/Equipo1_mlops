import logging
import visualization
import io

logger = logging.getLogger(__name__)


def visualizacion_inicial(df, df_org):
    logger.info("Inicia la impresión de la visualización inicial")

    logger.debug("\n" + "/"*50)
    logger.debug("VISUALIZACIÓN DE DATOS INICIAL")
    logger.debug("/"*50)

    # Mostrar forma y columnas
    logger.debug("\n" + "/"*50)
    logger.debug("DIMENSIONES Y COLUMNAS DEL DATASET")
    logger.debug("/"*50)
    logger.debug(f"Shape: {df.shape}")
    logger.debug(f"Columnas: {list(df.columns)}")

    # Información general
    logger.debug("\n" + "/"*50)
    logger.debug("INFORMACIÓN GENERAL DEL DATASET")
    logger.debug("/"*50)
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.debug(buffer.getvalue())

    # Primeras filas
    logger.debug("\n" + "/"*50)
    logger.debug("VISTA PREVIA")
    logger.debug("/"*50)
    logger.debug(f"\n{df.head()}")

    # Conteo de valores faltantes
    logger.debug("\n" + "/"*50)
    logger.debug("VALORES FALTANTES POR COLUMNA")
    logger.debug("/"*50)
    logger.debug(f"\n{df.isnull().sum()}")

    # Comparación datasets
    logger.debug("\n" + "/"*50)
    logger.debug("COMPARACIÓN DATASETS (MODIFICADO Y ORIGINAL)")
    logger.debug("/"*50)
    logger.debug(f"Modificado: {df.shape} | Original: {df_org.shape}")

    # Diferencias entre dataset
    extra_cols = df.columns.difference(df_org.columns)
    missing_cols = df_org.columns.difference(df.columns)

    logger.debug("\n" + "/"*50)
    logger.debug("COLUMNAS EXTRAS Y/O FALTANTES")
    logger.debug("/"*50)
    logger.debug(f"Extras en modificado: {list(extra_cols)} | Faltantes en modificado: {list(missing_cols)}")

    # Valores en Class
    logger.debug("\n" + "/"*50)
    logger.debug("VALORES EN CLASS")
    logger.debug("/"*50)
    logger.debug(f"{visualization.unique_class(df)}")

    # Tipos de datos
    logger.debug("\n" + "/"*50)
    logger.debug("COLUMNAS TIPO OBJECT")
    logger.debug("/"*50)
    obj_cols = df.select_dtypes(include="object").columns
    logger.debug(f"Total columnas tipo object: {len(obj_cols)} | Columnas: {obj_cols.tolist()}")
    logger.debug("Tenemos 36 columnas tipo object que deberían ser float64 - La columna Class debe permanecer como object")

    # --------------------------
    # Analisis de valores null
    # --------------------------
    logger.debug("\n" + "/"*50)
    logger.debug("ANÁLISIS DE VALORES NULL")
    logger.debug("/"*50)
    total_celdas = visualization.size_df(df)
    total_nulls = visualization.cantidad_nulls(df)

    logger.debug(f"Total de celdas: {total_celdas:,}")
    logger.debug(f"Total de valores nulos: {total_nulls:,} ({((total_nulls / total_celdas) * 100):.2f}%)")

    # Filas completas
    filas_completas = df.dropna().shape[0]
    logger.debug(f"Filas completamente válidas (sin NaN): {filas_completas}")

    diff = df.shape[0] - filas_completas
    logger.debug(f"Diferencia (filas incompletas): {diff}")

    logger.debug(
        "La cantidad de NaN es baja, pero están dispersos. "
        "No conviene eliminar filas, sino imputar valores según distribución."
    )

    logger.info("Termina la impresión de la visualización inicial")


def EDA(df):
    logger.info("Inicia la impresión de los datos EDA")

    logger.debug("\n" + "/"*50)
    logger.debug("ANÁLISIS EXPLORATORIO DE DATOS")
    logger.debug("/"*50 + "\n")

    cols_id = visualization.detectar_columnas_id(df)

    # Esta línea fallaba si el mock no devolvía tupla
    numeric_cols, cols_cat = visualization.variables_num_cat(df)

    # 1. Resumen general por columnas
    resumen = visualization.resumen_dataset(df)
    logger.debug("== Resumen por columna (ordenado por porcentaje de nulos) ==")
    logger.debug(f"\n{resumen.head(20)}")

    # 2. Columnas ID y objetivo
    logger.debug(f"\nColumnas con posibles ID: {visualization.detectar_columnas_id(df)}")

    objetivo = visualization.detectar_posible_objetivo(df)
    logger.debug(f"\nPosible columna objetivo: {objetivo}")

    # 3. Características descriptivas
    desc_num, desc_cat = visualization.estadisticas_descriptivas(df, excluir=cols_id)

    logger.debug("\n== Numéricas ==")
    logger.debug(f"\n{desc_num.head(20)}")

    logger.debug("\n== Categóricas (cardinalidad & nulos) ==")
    logger.debug(f"\n{desc_cat.head(20)}")

    # Distribuciones numéricas
    logger.debug("\nDistribuciones: Numéricas")
    visualization.distribuciones_numericas(df, numeric_cols)

    # Distribuciones categóricas
    logger.debug("\nDistribuciones: Categóricas")
    visualization.distribuciones_categoricas(df, columnas=cols_cat, top=10, max_columnas=9)

    # Correlación
    logger.debug("\nCorrelación")
    visualization.matriz_correlacion(df, excluir=visualization.detectar_columnas_id(df))

    # Outliers
    logger.debug("\nOutliers")
    visualization.boxplots_outliers_previos(numeric_cols, df)

    logger.info("Termina la impresión de los datos EDA")

    return cols_id
