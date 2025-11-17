import pandas as pd
import pytest
from src.data.load import carga_datos
from src.data.preprocessing import (
    eliminar_filas_invalidas,
    preprocesamiento_datos
)


def test_integration_load_to_preprocessing(tmp_path):
    # -------------------------------
    # 1) Crear archivos CSV temporales
    # -------------------------------
    modificado = tmp_path / "modificado.csv"
    original = tmp_path / "original.csv"

    df_mod = pd.DataFrame({
        "col1": [1, 2, None],
        "col2": ["a", "b", "c"]
    })

    df_orig = pd.DataFrame({
        "col1": [10, None, 30],
        "col2": ["x", "y", "z"]
    })

    df_mod.to_csv(modificado, index=False)
    df_orig.to_csv(original, index=False)

    # -------------------------------
    # 2) Cargar datos con load.py
    # -------------------------------
    df1, df2 = carga_datos(modificado, original)

    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert len(df1) == 3
    assert len(df2) == 3

    # -------------------------------
    # 3) Procesar datos con preprocessing.py
    # -------------------------------
    df1_clean = eliminar_filas_invalidas(df1.copy())
    df2_processed = preprocesamiento_datos(df2.copy())

    # -------------------------------
    # 4) Validar integración exitosa
    # -------------------------------
    # df1 tenía un None → debe eliminarse al limpiar
    assert len(df1_clean) == 2

    # df2_processed debe mantenerse como DataFrame válido
    assert isinstance(df2_processed, pd.DataFrame)

    # Debe tener las mismas columnas
    assert set(df2_processed.columns) == set(df2.columns)
