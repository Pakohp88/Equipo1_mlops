import pandas as pd
import numpy as np
import pytest
from data.preprocessing import (
    PreprocesamientoDF,
    preprocesamiento_datos,
    eliminar_filas_invalidas,
    reporte_nulos,
    imputacion_nulls,
    normalizacion,
    analisis_componentes_PCA,
)

# -----------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------

@pytest.fixture
def df_base():
    return pd.DataFrame({
        "A": [1, 2, None, 4],
        "B": ["10", "20", "xx", None],
        "Class": ["Pos", " Neg ", None, "pos"],
        "mixed_type_col": [1, 2, 3, 4]
    })


# -----------------------------------------------------------
# TEST PreprocesamientoDF
# -----------------------------------------------------------

def test_eliminar_columnas(df_base):
    pre = PreprocesamientoDF(df_base.copy())
    pre.eliminar_columnas(["mixed_type_col"])
    assert "mixed_type_col" not in pre.df.columns


def test_convertir_tipo(df_base):
    pre = PreprocesamientoDF(df_base.copy())
    pre.convertir_tipo()

    # B debe ser numérica luego de convertir ("xx" →
