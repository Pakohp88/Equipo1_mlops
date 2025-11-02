import logging
import visualization

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
  logger.debug(df.shape)
  logger.debug(list(df.columns))

  # Tipos de datos y valores nulos (vista rápida)
  logger.debug("\n" + "/"*50)
  logger.debug("INFORMACIÓN GENERAL DEL DATASET")
  logger.debug("/"*50)
  logger.debug(df.info())

  # Primeras filas
  logger.debug("\n" + "/"*50)
  logger.debug("VISTA PREVIA")
  logger.debug("/"*50)
  logger.debug(df.head())

  # Conteo de valores faltantes
  logger.debug("\n" + "/"*50)
  logger.debug("VALORES FALTANTES POR COLUMNA")
  logger.debug("/"*50)
  logger.debug(df.isnull().sum())

  # Comparación datasets (modificado y original)
  logger.debug("\n" + "/"*50)
  logger.debug("COMPARACIÓN DATASETS (MODIFICADO Y ORIGINAL)")
  logger.debug("/"*50)
  logger.debug("Modificado:", df.shape, "Original:", df_org.shape)

  # Diferencias entre dataset
  extra_cols = df.columns.difference(df_org.columns)
  missing_cols = df_org.columns.difference(df.columns)
  logger.debug("\n" + "/"*50)
  logger.debug("COLUMNAS EXTRAS Y/O FALTANTES")
  logger.debug("/"*50)
  logger.debug("Extras en modificado:", list(extra_cols), "faltantes en modificado:", list(missing_cols))

  logger.debug("\n" + "/"*50)
  logger.debug("VALORES EN CLASS")
  logger.debug("/"*50)  
  logger.debug(visualization.unique_class(df))

  # Tipos de datos
  logger.debug("\n" + "/"*50)
  logger.debug("COLUMNAS TIPO OBJECT")
  logger.debug("/"*50)  
  logger.debug("Total columnas tipo object:", len(df.select_dtypes(include="object").columns), "Columnas:", (df.select_dtypes(include="object").columns).tolist())
  logger.debug("Tenemos 36 columnas tipo object que deberían ser float64 - La columna Class debe permanecer como object")

  # --------------------------
  # Analisis de valores null
  # --------------------------
  logger.debug("\n" + "/"*50)
  logger.debug("ANÁLISIS DE VALORES NULL")
  logger.debug("/"*50)
  logger.debug(f"Total de celdas: {visualization.size_df(df):,}")
  logger.debug(f"Total de valores nulos: {visualization.cantidad_nulls(df):,}  ({((visualization.cantidad_nulls(df) / visualization.size_df(df)) * 100):.2f}%)")

  # Contar filas completas sin ningun NaN
  filas_completas = df.dropna().shape[0]
  logger.debug("Filas completamente validas (sin ningun NaN):", filas_completas)

  # Y posteriormente, comparamos contra el original para ver la diferencia
  diff = df.shape[0] - filas_completas
  logger.debug("Diferencia (filas incompletas):", diff)

  logger.debug("""La cantidad de NaN es bajo, pero estan dispersos ya que la aproximadamente la mitad de las filas (213) tienen valores nulos
  Por eso no conviene eliminar todas esas filas, sino reparar los valores\n
  Si los valores no nulos de una columna tienen una distribucion normal (sin valores extremos grandes), entonces el promedio representaria
  bien a la columna. Pero si la columna tiene valores muy dispersos, el promedio puede distorsionar. La mediana representa mejor los
  valores centrales para rellenar los nulos, porque no se ve afectada por outliers
  """)

  logger.info("Termina la impresión de la visualización inicial")

def EDA(df):
  logger.info("Inicia la impresión de los datos EDA")

  logger.debug("\n" + "/"*50)
  logger.debug("ANÁLISIS EXPLORATORIO DE DATOS")
  logger.debug("/"*50 + "\n")
  cols_id = visualization.detectar_columnas_id(df)
  numeric_cols, cols_cat = visualization.variables_num_cat(df)

  # 1. Resumen general por columnas
  resumen = visualization.resumen_dataset(df)
  logger.debug("== Resumen por columna (ordenado por porcentaje de nulos) ==")
  logger.debug(resumen.head(20))

  # 2. Identificación de columnas ID y Objetivo
  logger.debug("\nColumnas con posibles de ID:", visualization.detectar_columnas_id(df))

  objetivo = visualization.detectar_posible_objetivo(df)
  logger.debug("\nPosible columna objetivo:", objetivo)

  # 3. Características descriptivas
  desc_num, desc_cat = visualization.estadisticas_descriptivas(df, excluir=cols_id)
  logger.debug("\n== Numéricas ==")
  logger.debug(desc_num.head(20))
  logger.debug("\n== Categóricas (cardinalidad & nulos) ==")
  logger.debug(desc_cat.head(20))

  # 2. Distribuciones numéricas
  logger.debug("\nDistribuciones: Numéricas")
  visualization.distribuciones_numericas(df, numeric_cols)

  # 3. Distribuciones categóricas
  logger.debug("\nDistribuciones: Categóricas")
  visualization.distribuciones_categoricas(df, columnas=cols_cat, top=10, max_columnas=9)

  # 4. Correlación de variables
  logger.debug("\nCorrelación")
  visualization.matriz_correlacion(df, excluir=visualization.detectar_columnas_id(df))

  # 5. Outliers
  logger.debug("\nOutliers")
  visualization.boxplots_outliers_previos(numeric_cols, df)

  logger.info("Termina la impresión de los datos EDA")

  return cols_id