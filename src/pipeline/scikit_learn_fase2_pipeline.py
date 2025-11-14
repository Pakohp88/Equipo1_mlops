import pandas as pd
import logging
import yaml
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar configuración desde config.yaml
CONFIG_PATH = Path("config.yaml")
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

CLEAN_DATA = Path(config['paths']['clean_data'])   
PARAMS_RUNS = config['experimentos']['sl_1']

# cargar datos
df = pd.read_csv(CLEAN_DATA)

# separar caracteristicas y objetivo
X = df.drop(columns=["Class"])
y = df["Class"]

# separar variables numéricas
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# preprocesamiento y transformaciones
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols)
])

# definir nombre de experimento
mlflow.set_experiment("Basic_RandomForest_Classifier_Experiment")
for i, params in enumerate(PARAMS_RUNS):
    with mlflow.start_run(run_name=f"run_{i+1}") as run:
        # Log de parámetros
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        # Pipeline completo
        model_pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42))
        ])

        # separar datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # entrenamiento
        model_pipeline.fit(X_train, y_train)

        # evaluación
        y_pred = model_pipeline.predict(X_test)
        logging.info(classification_report(y_test, y_pred))

        # metricas
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # registro en ML flow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        # validación cruzada
        scores = cross_val_score(model_pipeline, X, y, cv=5)
        logging.info(f"Accuracy promedio (CV): {scores.mean():.2f}")
        
        # Guardar el modelo
        mlflow.sklearn.log_model(model_pipeline, "scikit_learn_fase2_1")

        run_id = run.info.run_id

        # registra el modelo
        model_uri = f"runs:/{run_id}/scikit_learn_fase2_1"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="fase2_model"
        )

