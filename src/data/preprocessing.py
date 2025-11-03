import numpy as np
import pandas as pd
import re
import logging
import visualization
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer

logger = logging.getLogger(__name__)

class PreprocesamientoDF:
  def __init__(self, df):
    self.df = df

    logger.debug("\n" + "/"*50)
    logger.debug("PRE-PROCESAMIENTO DE DATOS")
    logger.debug("/"*50 + "\n")

  def eliminar_columnas(self, columnas):
    self.df.drop(columns=columnas, inplace=True)
    return self

  def limpieza_class(self):
    # homogenizar los valores
    self.df["Class"] = self.df["Class"].str.strip().str.lower()
    # Remplaza los valores NaN con la moda
    moda_class = self.df["Class"].mode()[0]
    self.df["Class"] = self.df["Class"].fillna(moda_class)

  def tratamiento_nulls(self):
    numeric_cols = self.df.select_dtypes(include="float64").columns
    self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
    return self

  # Manejo de valores no numéricos (object) en columnas que deberían ser numéricas (float64)
  def convertir_tipo(self):
    invalid_tokens_set = set() # Para recolectar los valores no numéricos en las columnas
    obj_cols = self.df.select_dtypes(include="object").columns

    for col in obj_cols:
      if col == "Class":
        continue

      # Eliminar caracteres válidos de números (0-9, punto, signo, notación científica)
      s = self.df[col].dropna().astype(str).str.strip()
      non_numeric = s[~s.str.replace(r"[0-9\.\-eE]", "", regex=True).eq("")]
      if not non_numeric.empty:
        invalid_tokens_set.update(non_numeric.unique())

      # Reemplazar invalid_tokens_set con np.nan y convertir a float
      self.df[col] = self.df[col].replace(non_numeric.unique(), np.nan)
      self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

  def analisis_outliers(self):
    numeric_cols = self.df.select_dtypes(include=["float64"]).columns
    outlier_summary = []

    for col in numeric_cols:
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
        percent = outliers / len(self.df) * 100
        outlier_summary.append((col, outliers, percent))
    return outlier_summary

  def normalizar_nombres_columnas(self: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres (snake_case, sin acentos). No altera datos."""
    def norm(s: str) -> str:
        s = s.strip().lower()
        s = (s.replace("á","a").replace("é","e").replace("í","i")
              .replace("ó","o").replace("ú","u").replace("ñ","n"))
        s = re.sub(r'[^a-z0-9]+', '_', s)
        s = re.sub(r'_+','_', s).strip('_')
        return s
    return self.df.rename(columns={c: norm(c) for c in self.df.columns})

  def limpiar_textos(self: pd.DataFrame) -> pd.DataFrame:
    obj_cols = [c for c in self.df.columns if self.df[c].dtype == "object"]
    for c in obj_cols:
        self.df[c] = self.df[c].astype(str).str.strip()
        self.df[c] = self.df[c].replace({"": np.nan, "nan": np.nan, "None": np.nan})

  # Recortar valores fuera de los limites [Q1 - factor*IQR, Q3 + factor*IQR]
  def winsorize_iqr(self, factor=1.5):
      numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns

      for col in numeric_cols:
          Q1 = self.df[col].quantile(0.25)
          Q3 = self.df[col].quantile(0.75)
          IQR = Q3 - Q1
          lower = Q1 - factor * IQR
          upper = Q3 + factor * IQR
          self.df[col] = np.clip(self.df[col], lower, upper)

def preprocesamiento_datos(df):
  logger.info("Inicial el preprocesamiento de datos")

  df_limpiando = PreprocesamientoDF(df)

  # 1. Eliminar columna extra: mixex_type_col
  df_limpiando.eliminar_columnas(['mixed_type_col'])

  # 2. TRATAMIENTO DE VALORES INCONSISTENTES (OBJECT - NULL)
  logger.debug("\n")
  df_limpiando.convertir_tipo()
  logger.debug("Tipos de datos después de aplicar la conversión")
  logger.debug(df.dtypes.value_counts())

  # 3. TRATAMIENTO DE VALORES NULL
  logger.debug("\n")
  df_limpiando.tratamiento_nulls()

  # 4. TRATAMIENTO CLASS
  logger.debug("\n")
  df_limpiando.limpieza_class()

  # 5. Visualización de outliers
  logger.debug("\n")
  outlier_summary = df_limpiando.analisis_outliers()
  outlier_df = pd.DataFrame(outlier_summary, columns=["columna", "n_outliers", "porcentaje"])
  outlier_df = outlier_df.sort_values("porcentaje", ascending=False)
  logger.debug("Columnas con más OUTLIERS:")
  logger.debug(outlier_df)

  # 5.1 Tratamiento de outliers
  df_limpiando.winsorize_iqr()

  # 6. Normalizar nombre de columnas (facilitar análisis - no cambia datos)
  df_limpiando.normalizar_nombres_columnas()
  df_limpiando.limpiar_textos()

  logger.info("Finaliza el preprocesamiento de datos")

def eliminar_filas_invalidas(df):
  col_objetivo = "class"
  filas_antes = len(df)
  if col_objetivo in df.columns:
      df = df[~df[col_objetivo].isna()].copy()
      logger.debug(f"✔ Filas sin objetivo eliminadas: {filas_antes - len(df)}")

  dup = int(df.duplicated().sum())
  df = df.drop_duplicates().copy()
  logger.debug("✔ Filas duplicadas eliminadas:", dup)

def reporte_nulos(df):
  tabla_nulos = (
    df.isna().sum().to_frame("nulos")
      .assign(porcentaje=lambda t: (t["nulos"]/len(df)*100).round(2))
      .sort_values("nulos", ascending=False)
  )

  return tabla_nulos

def imputacion_nulls(df):
  #IMPUTACIÓN DE NULOS (mediana / moda)
  num_cols = [c for c in df.select_dtypes(include=np.number).columns]
  cat_cols = [c for c in df.select_dtypes(exclude=np.number).columns]

  # Medianas para numéricas
  medianas = df[num_cols].median(numeric_only=True)
  df[num_cols] = df[num_cols].fillna(medianas)

  # Modas para categóricas
  modas = {}
  for c in cat_cols:
      moda = df[c].mode(dropna=True)
      modas[c] = (moda.iloc[0] if not moda.empty else "desconocido")
      df[c] = df[c].fillna(modas[c])

  # PENDIENTE
  # Guardar diccionarios de imputación (para reproducibilidad)
  # (DIR_META / "imputacion_medianas.json").write_text(medianas.to_json(), encoding="utf-8")
  # (DIR_META / "imputacion_modas.json").write_text(json.dumps(modas, ensure_ascii=False, indent=2), encoding="utf-8")

  logger.debug("Imputación realizada (medianas y modas)")

def normalizacion(df):
  numeric_cols, cols_cat = visualization.variables_num_cat(df)
  scaler = PowerTransformer(method='yeo-johnson')
  df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

def analisis_componentes_PCA(df):
  numeric_cols, cols_cat = visualization.variables_num_cat(df)
  pca = PCA()
  df_numeric = df[numeric_cols]
  pca.fit(df_numeric)
  explained_variance = pca.explained_variance_ratio_
  return explained_variance

def limpieza_preparacion(df):
  logger.info("Inicial limpieza y preparación de datos")

  logger.debug("\n" + "/"*50)
  logger.debug("LIMPIEZA Y PREPARACIÓN")
  logger.debug("/"*50 + "\n")

  # 1.
  logger.debug("Eliminar filas inválidas del objetivo y duplicados")
  eliminar_filas_invalidas(df)
  # 2.
  logger.debug("\nReporte de nulos")
  tabla_nulos_antes = reporte_nulos(df)
  # PENDIENTE
  #tabla_nulos_antes.to_csv(DIR_META / "nulos_antes.csv", encoding="utf-8")
  logger.debug("Top nulos (antes):")
  tabla_nulos_antes.head(15)
  # 3.
  logger.debug("\nImputación (mediana para numpericas, moda para categóricas)")
  imputacion_nulls(df)
  # 4.
  logger.debug("\nNormalización de Datos")
  normalizacion(df)
  df.head()
  # 5.
  logger.debug("\nAnálisis de Componentes")
  explained_variance = analisis_componentes_PCA(df)
  # 6.
  logger.debug("\n")
  visualization.grafica_analisis_componentes_PCA(explained_variance)
  # 7.
  logger.debug("\nReporte de nulos - posterior")
  tabla_nulos_despues = reporte_nulos(df)
  # PENDIENTE
  #tabla_nulos_despues.to_csv(DIR_META / "nulos_despues.csv", encoding="utf-8")
  logger.debug("Nulos (después) - top:")
  tabla_nulos_despues.head(10)
  # 8. PENDIENTE
  # Guardar CSV/parquet limpios
  # RUTA_LIMPIO_CSV = DIR_CLEAN / "dataset_limpio.csv"
  # RUTA_LIMPIO_PAR = DIR_CLEAN / "dataset_limpio.parquet"
  # df.to_csv(RUTA_LIMPIO_CSV, index=False, encoding="utf-8")
  # df.to_parquet(RUTA_LIMPIO_PAR, index=False)

  # print("✅ Dataset limpio guardado en:")
  # print("  -", RUTA_LIMPIO_CSV)
  # print("  -", RUTA_LIMPIO_PAR)
  logger.info("Finaliza limpieza y preparación de datos")