import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


# ============================================================
# 1. Separación X y y
# ============================================================
def separar_x_y(df):
    col_objetivo = "class"
    if col_objetivo in df.columns:
        y = df[col_objetivo]
        X = df.drop(columns=[col_objetivo])
        logger.info("Columnas separadas correctamente.")
        return X, y
    else:
        logger.warning("⚠ No existe la columna 'class' en el dataframe.")
        return None, None


# ============================================================
# 2. Cargar X y y procesados
# ============================================================
def obtener_X_y():
    try:
        X = pd.read_csv("data/processed/X.csv")
        y = pd.read_csv("data/processed/y.csv").squeeze()
        return X, y
    except FileNotFoundError:
        logger.error("No se encontraron los archivos X.csv o y.csv.")
        return None, None


# ============================================================
# 3. Partición del dataset
# ============================================================
def particion_dataSet(X, y):
    X_temp, X_validation, y_temp, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    return X_train, X_test, y_train, y_test, X_validation, y_validation


# ============================================================
# 4. Entrenamiento de modelos
# ============================================================
def train_model(X_train, y_train):
    log_reg = LogisticRegression(max_iter=500)
    dec_tree = DecisionTreeClassifier()
    rand_forest = RandomForestClassifier()
    svm_model = SVC()
    knn_model = KNeighborsClassifier()
    gradient_boosting = GradientBoostingClassifier()

    log_reg.fit(X_train, y_train)
    dec_tree.fit(X_train, y_train)
    rand_forest.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)
    gradient_boosting.fit(X_train, y_train)

    return log_reg, dec_tree, rand_forest, svm_model, knn_model, gradient_boosting


# ============================================================
# 5. Hiperparámetros
# ============================================================
def ajuste_hiperparametros():
    param_grid_log_reg = {"C": [0.1, 1], "solver": ["liblinear"]}
    param_grid_dec_tree = {"max_depth": [None, 10]}
    param_grid_rand_forest = {"n_estimators": [50, 100]}
    param_grid_svm = {"C": [0.1, 1], "kernel": ["linear"]}
    param_grid_knn = {"n_neighbors": [3, 5]}
    param_grid_gradient_boosting = {"n_estimators": [50, 100]}

    resumen = pd.DataFrame([
        {"Modelo": "Logistic Regression", "Hiperparámetros": str(param_grid_log_reg)},
        {"Modelo": "Decision Tree", "Hiperparámetros": str(param_grid_dec_tree)},
        {"Modelo": "Random Forest", "Hiperparámetros": str(param_grid_rand_forest)},
        {"Modelo": "SVM", "Hiperparámetros": str(param_grid_svm)},
        {"Modelo": "KNN", "Hiperparámetros": str(param_grid_knn)},
        {"Modelo": "Gradient Boosting", "Hiperparámetros": str(param_grid_gradient_boosting)},
    ])

    return (
        param_grid_log_reg,
        param_grid_dec_tree,
        param_grid_rand_forest,
        param_grid_svm,
        param_grid_knn,
        param_grid_gradient_boosting,
        resumen,
    )


# ============================================================
# 6. GridSearch
# ============================================================
def aplicar_gridSearch(estimator, params):
    return GridSearchCV(estimator, params, cv=3)


def fit_gridSearch(X_train, y_train, grid_search):
    grid_search.fit(X_train, y_train)
    return grid_search


def store_estimators(grid_search):
    return grid_search.best_estimator_


# ============================================================
# 7. Pipeline de evaluación (opcional)
# ============================================================
def evaluar_modelos():
    logger.info("Inicia evaluación")

    X, y = obtener_X_y()
    if X is None or y is None:
        return None

    X_train, X_test, y_train, y_test, X_val, y_val = particion_dataSet(X, y)

    modelos = train_model(X_train, y_train)

    (
        p1, p2, p3, p4, p5, p6, tabla
    ) = ajuste_hiperparametros()

    param_grids = [p1, p2, p3, p4, p5, p6]

    gridsearches = [
        aplicar_gridSearch(modelos[i], param_grids[i])
        for i in range(len(modelos))
    ]

    entrenados = [fit_gridSearch(X_train, y_train, gs) for gs in gridsearches]

    best_estimators = [store_estimators(gs) for gs in entrenados]

    return best_estimators
