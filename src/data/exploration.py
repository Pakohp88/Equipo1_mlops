import logging

def visualizacion_inicial(df, df_org):
  print("\n")
  print("/"*50)
  print("VISUALIZACIÓN DE DATOS INICIAL")
  print("/"*50)

  # Mostrar forma y columnas
  print("\n")
  print("="*50)
  print("DIMENSIONES Y COLUMNAS DEL DATASET")
  print("="*50)
  print(df.shape)
  print(list(df.columns))

  # Tipos de datos y valores nulos (vista rápida)
  print("\n")
  print("="*50)
  print("INFORMACIÓN GENERAL DEL DATASET")
  print("="*50)
  print(df.info())

  # Primeras filas
  print("\n")
  print("="*50)
  print("VISTA PREVIA")
  print("="*50)
  print(df.head())

  # Conteo de valores faltantes
  print("\n")
  print("="*50)
  print("VALORES FALTANTES POR COLUMNA")
  print("="*50)
  print(df.isnull().sum())

  # Comparación datasets (modificado y original)
  print("\n")
  print("="*50)
  print("COMPARACIÓN DATASETS (MODIFICADO Y ORIGINAL)")
  print("="*50)
  print("Modificado:", df.shape, "Original:", df_org.shape)

  # Diferencias entre dataset
  extra_cols = df.columns.difference(df_org.columns)
  missing_cols = df_org.columns.difference(df.columns)
  print("\n")
  print("="*50)
  print("COLUMNAS EXTRAS Y/O FALTANTES")
  print("="*50)
  print("Extras en modificado:", list(extra_cols), "faltantes en modificado:", list(missing_cols))

  print("\n")
  print("="*50)
  print("VALORES EN CLASS")
  print("="*50)
  unique_class(df)

  # Tipos de datos
  print("\n")
  print("="*50)
  print("COLUMNAS TIPO OBJECT")
  print("="*50)
  print("Total columnas tipo object:", len(df.select_dtypes(include="object").columns), "Columnas:", (df.select_dtypes(include="object").columns).tolist())
  print("Tenemos 36 columnas tipo object que deberían ser float64 - La columna Class debe permanecer como object")

  # --------------------------
  # Analisis de valores null
  # --------------------------
  print("\n")
  print("="*50)
  print("ANÁLISIS DE VALORES NULL")
  print("="*50)
  print(f"Total de celdas: {size_df(df):,}")
  print(f"Total de valores nulos: {cantidad_nulls(df):,}  ({((cantidad_nulls(df) / size_df(df)) * 100):.2f}%)")

  # Contar filas completas sin ningun NaN
  filas_completas = df.dropna().shape[0]
  print("Filas completamente validas (sin ningun NaN):", filas_completas)

  # Y posteriormente, comparamos contra el original para ver la diferencia
  diff = df.shape[0] - filas_completas
  print("Diferencia (filas incompletas):", diff)

  print("""La cantidad de NaN es bajo, pero estan dispersos ya que la aproximadamente la mitad de las filas (213) tienen valores nulos
  Por eso no conviene eliminar todas esas filas, sino reparar los valores\n
  Si los valores no nulos de una columna tienen una distribucion normal (sin valores extremos grandes), entonces el promedio representaria
  bien a la columna. Pero si la columna tiene valores muy dispersos, el promedio puede distorsionar. La mediana representa mejor los
  valores centrales para rellenar los nulos, porque no se ve afectada por outliers
  """)

def EDA(df):
  print("\n")
  print("/"*50)
  print("ANÁLISIS EXPLORATORIO DE DATOS")
  print("/"*50)
  print("\n")

  cols_id = detectar_columnas_id(df)
  numeric_cols, cols_cat = variables_num_cat(df)

  # 1. Resumen general por columnas
  resumen = resumen_dataset(df)
  print("== Resumen por columna (ordenado por % de nulos) ==")
  print(resumen.head(20))

  # 2. Identificación de columnas ID y Objetivo
  print("\nColumnas con posibles de ID:", detectar_columnas_id(df))

  objetivo = detectar_posible_objetivo(df)
  print("\nPosible columna objetivo:", objetivo)

  # 3. Características descriptivas
  desc_num, desc_cat = estadisticas_descriptivas(df, excluir=cols_id)
  print("\n== Numéricas ==")
  print(desc_num.head(20))
  print("\n== Categóricas (cardinalidad & nulos) ==")
  print(desc_cat.head(20))

  # 2. Distribuciones numéricas
  print("\nDistribuciones: Numéricas")
  distribuciones_numericas(df, numeric_cols)

  # 3. Distribuciones categóricas
  print("\nDistribuciones: Categóricas")
  distribuciones_categoricas(df, columnas=cols_cat, top=10, max_columnas=9)

  # 4. Correlación de variables
  print("\nCorrelación")
  matriz_correlacion(df, excluir=detectar_columnas_id(df))

  # 5. Outliers
  print("\nOutliers")
  boxplots_outliers_previos(numeric_cols, df)

  return cols_id