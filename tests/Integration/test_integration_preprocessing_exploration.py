import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.data.preprocessing import eliminar_filas_invalidas
from src.data.exploration import visualizacion_inicial, EDA


# ================================================================
# TEST INTEGRADO: preprocessing → exploration
# ================================================================
def test_preprocessing_to_exploration():

    # ---------------------------------------
    # 1) Dataset inicial con valores nulos
    # ---------------------------------------
    df = pd.DataFrame({
        "col1": [1, None, 3],
        "col2": ["a", "b", "c"],
        "class": ["yes", "no", "yes"]
    })

    df_org = df.copy()

    # ---------------------------------------
    # 2) Aplicar limpieza de preprocessing
    # ---------------------------------------
    df_clean = eliminar_filas_invalidas(df)

    # Debe eliminar 1 fila por NaN
    assert len(df_clean) == 2

    # ---------------------------------------
    # 3) Mockear el módulo visualization
    # ---------------------------------------
    with patch("src.data.exploration.visualization") as mock_vis:

        # Configurar retornos para evitar errores
        mock_vis.unique_class.return_value = ["yes", "no"]
        mock_vis.size_df.return_value = 6
        mock_vis.cantidad_nulls.return_value = 0
        mock_vis.detectar_columnas_id.return_value = []
        mock_vis.variables_num_cat.return_value = (["col1"], ["col2"])
        mock_vis.resumen_dataset.return_value = pd.DataFrame({"col": ["col1", "col2"]})
        mock_vis.estadisticas_descriptivas.return_value = (
            pd.DataFrame({"a": [1, 2]}),
            pd.DataFrame({"b": [3, 4]})
        )

        # Funciones que generan gráficos → se mockean para no fallar
        mock_vis.distribuciones_numericas.return_value = None
        mock_vis.distribuciones_categoricas.return_value = None
        mock_vis.matriz_correlacion.return_value = None
        mock_vis.boxplots_outliers_previos.return_value = None

        # ---------------------------------------
        # 4) Ejecutar las funciones de exploration
        # ---------------------------------------
        visualizacion_inicial(df_clean, df_org)

        cols_id = EDA(df_clean)

        # Validar que exploration devolvió algo válido
        assert isinstance(cols_id, list)
