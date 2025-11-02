import pandas as pd
import logging
import yaml
import mlflow
import mlflow.sklearn


from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar configuración desde config.yaml
CONFIG_PATH = Path("config.yaml")
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

CLEAN_DATA = Path(config['paths']['clean_data'])    


# definir nombre de experimento
mlflow.set_experiment("Basic_RandomForest_Classifier_Experiment")


with mlflow.start_run():
    # Log de parámetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)

    # cargar datos
    df = pd.read_csv(CLEAN_DATA)

    # separar caracteristicas y objetivo
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # preprocesamiento

    # transformaciones

    # Pipeline completo
    model_pipeline = Pipeline(steps=[
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # separar datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # entrenamiento
    model_pipeline.fit(X_train, y_train)

    # evaluación
    y_pred = model_pipeline.predict(X_test)
    logging.info(classification_report(y_test, y_pred))

    # validación cruzada
    scores = cross_val_score(model_pipeline, X, y, cv=5)
    logging.info(f"Accuracy promedio (CV): {scores.mean():.2f}")

    
    # Guardar el modelo
    mlflow.sklearn.log_model(model_pipeline, "scikit_learn_fase2_1")

