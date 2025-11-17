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
        columnas_existentes = [c for c in columnas if c in self.df.columns]
        if columnas_existentes:
            self.df.drop(columns=columnas_existentes, inplace=True)

    def limpieza_class(self):
        if "Class" in self.df.columns:
            self.df["Class"] = self.df["Class"].astype(str).str.strip().str.lower()

    def tratamiento_nulls(self):
        numeric_cols = self.df.select_dtypes(include="float64").columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        return self

    # ---- FIX aplicado AQUÍ ----
    def convertir_tipo(self):
        invalid_tokens_set = set()
        obj_cols = self.df.select_dtypes(include="object").columns

        for col in obj_cols:
            if col == "Class":
                continue

            s = self.df[col].dropna().astype(str).str.strip()
            non_numeric = s[~s.str.replace(r"[0-9\.\-eE]", "", regex=True).eq("")]

            if not non_numeric.empty:
                invalid_tokens_set.update(non_numeric.unique())

            # FIX → evitar FutureWarning aplicando infer_objects(copy=False)
            self.df[col] = (
                self.df[col]
                .replace(non_numeric.unique(), np.nan)
                .infer_objects(copy=False)
            )

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

    df_limpiando.eliminar_columnas(['mixed_type_col'])
    df_limpiando.convertir_tipo()

    logger.debug("Tipos de datos después de aplicar la conversión")
    logger.debug(df.dtypes.value_counts())

    df_limpiando.tratamiento_nulls()
    df_limpiando.limpieza_class()

    outlier_summary = df_limpiando.analisis_outliers()
    outlier_df = pd.DataFrame(outlier_summary, columns=["columna", "n_outliers", "porcentaje"])
    outlier_df = outlier_df.sort_values("porcentaje", ascending=False)

    logger.debug("Columnas con más OUTLIERS:")
    logger.debug(outlier_df)

    df_limpiando.winsorize_iqr()

    df_limpiando.normalizar_nombres_columnas()
    df_limpiando.limpiar_textos()

    logger.info("Finaliza el preprocesamiento de datos")
    return df_limpiando.df


def eliminar_filas_invalidas(df):
    filas_antes = len(df)

    df = df.dropna().copy()
    logger.debug(f"✔ Filas con valores nulos eliminadas: {filas_antes - len(df)}")

    dup = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    logger.debug(f"✔ Filas duplicadas eliminadas: {dup}")

    return df

def reporte_nulos(df):
    tabla_nulos = (
        df.isna().sum().to_frame("nulos")
        .assign(porcentaje=lambda t: (t["nulos"]/len(df)*100).round(2))
        .sort_values("nulos", ascending=False)
    )
    return tabla_nulos

def imputacion_nulls(df):
    num_cols = [c for c in df.select_dtypes(include=np.number).columns]
    cat_cols = [c for c in df.select_dtypes(exclude=np.number).columns]

    medianas = df[num_cols].median(numeric_only=True)
    df[num_cols] = df[num_cols].fillna(medianas)

    modas = {}
    for c in cat_cols:
        moda = df[c].mode(dropna=True)
        modas[c] = (moda.iloc[0] if not moda.empty else "desconocido")
        df[c] = df[c].fillna(modas[c])

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
    return pca.explained_variance_ratio_

def limpieza_preparacion(df):
    logger.info("Inicial limpieza y preparación de datos")

    logger.debug("\n" + "/"*50)
    logger.debug("LIMPIEZA Y PREPARACIÓN")
    logger.debug("/"*50 + "\n")

    eliminar_filas_invalidas(df)

    tabla_nulos_antes = reporte_nulos(df)
    tabla_nulos_antes.head(15)

    imputacion_nulls(df)

    normalizacion(df)
    df.head()

    explained_variance = analisis_componentes_PCA(df)

    visualization.grafica_analisis_componentes_PCA(explained_variance)

    tabla_nulos_despues = reporte_nulos(df)
    tabla_nulos_despues.head(10)

    logger.info("Finaliza limpieza y preparación de datos")
