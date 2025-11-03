import load
import exploration
import preprocessing
import visualization
import training_testing

# -----------------------
# CARGA DE DATOS
# -----------------------
df, df_org = load.carga_datos('turkish_music_emotion_modified.csv','turkis_music_emotion_original.csv')

# -------------------------------
# VISUALIZACIÓN DE DATOS INICIAL
# -------------------------------
exploration.visualizacion_inicial(df, df_org)

# ---------------------------
# PRE-PROCESAMIENTO DE DATOS
# ---------------------------
preprocessing.preprocesamiento_datos(df)

# -----------------------
# 99. IMPRIME
# -----------------------
visualization.imprime_previo_eda(df)

# --------------------------------------
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# --------------------------------------
cols_id = exploration.EDA(df)

# --------------------------------------
# LIMPIEZA Y PREPARACIÓN
# --------------------------------------
preprocessing.limpieza_preparacion(df)

# -----------------------------------------------
# 99. IMPRIME POSTERIOR A PROCESOS DE LIMPIEZA
# -----------------------------------------------
print("\nValidación de estadísticas después de aplicar procesos de limpieza")
df.describe()
visualization.imprime_posterior_eda(df)

# --------------------------------------
# SEPARAR X y y
# --------------------------------------
training_testing.separar_x_y(df)

# PENDIENTE: AQUI HAY QUE CORRER LAS CELDAS DE PREPARAR AMBIENTE Y VERSIONADO DE DATOS

# --------------------------------------
# ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# --------------------------------------
training_testing.evaluar_modelos()

# PENDIENTE: SUBIR CAMBIOS A GITHUB