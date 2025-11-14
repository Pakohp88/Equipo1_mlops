import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from src.data.training_testing import (
    separar_x_y,
    particion_dataSet,
    train_model,
    ajuste_hiperparametros,
    aplicar_gridSearch,
    fit_gridSearch,
    store_estimators
)


# ---------------------------------------------------------------
# 1. separar_x_y
# ---------------------------------------------------------------
def test_separar_x_y_correcto():
    df = pd.DataFrame({
        "class": [0, 1, 0],
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    X, y = separar_x_y(df)

    assert X.shape == (3, 2)
    assert y.shape == (3,)


def test_separar_x_y_sin_columna_class():
    df = pd.DataFrame({"A": [1, 2]})
    X, y = separar_x_y(df)
    assert X is None and y is None


# ---------------------------------------------------------------
# 2. particion_dataSet
# ---------------------------------------------------------------
def test_particion_dataset():
    X = pd.DataFrame({
        "A": np.random.rand(100),
        "B": np.random.rand(100)
    })
    y = pd.Series([0, 1] * 50)

    X_train, X_test, y_train, y_test, X_val, y_val = particion_dataSet(X, y)

    assert len(X_train) == 60
    assert len(X_test) == 20
    assert len(X_val) == 20


# ---------------------------------------------------------------
# 3. train_model
# ---------------------------------------------------------------
def test_train_model_retorna_modelos():
    X = pd.DataFrame({"A": np.random.rand(50)})
    y = pd.Series([0, 1] * 25)

    models = train_model(X, y)
    assert len(models) == 6


# ---------------------------------------------------------------
# 4. ajuste_hiperparametros
# ---------------------------------------------------------------
def test_ajuste_hiperparametros():
    salida = ajuste_hiperparametros()
    assert len(salida) == 7  # 6 grids + tabla resumen
    tabla = salida[-1]
    assert isinstance(tabla, pd.DataFrame)
    assert tabla.shape[0] == 6


# ---------------------------------------------------------------
# 5. aplicar_gridSearch
# ---------------------------------------------------------------
def test_aplicar_gridsearch_instancia():
    estimator = LogisticRegression()
    params = {"C": [1, 10]}
    gs = aplicar_gridSearch(estimator, params)
    assert isinstance(gs, GridSearchCV)


# ---------------------------------------------------------------
# 6. fit_gridSearch
# ---------------------------------------------------------------
def test_fit_gridsearch_entrenado():
    X = pd.DataFrame({"A": np.random.rand(20)})
    y = pd.Series([0, 1] * 10)

    gs = aplicar_gridSearch(LogisticRegression(), {"C": [1]})
    gs = fit_gridSearch(X, y, gs)

    assert hasattr(gs, "best_estimator_")


# ---------------------------------------------------------------
# 7. store_estimators
# ---------------------------------------------------------------
def test_store_estimators_ok():
    class DummyGS:
        best_estimator_ = "OK"

    gs = DummyGS()
    best = store_estimators(gs)

    assert best == "OK"
