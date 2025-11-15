import pytest
import pandas as pd
import logging
from unittest.mock import patch

# Importación ajustada al ruteo de su proyecto (src/data/exploration.py)
from src.data import exploration 


# =============================================================
# FIXTURE DE DATOS SIMULADOS
# =============================================================
@pytest.fixture
def mock_data():
    """Genera un DataFrame de ejemplo simple con un nulo para las pruebas."""
    data = {
        'Feature1': [1, 2, 3, 4],
        'Feature2': [5.0, None, 7.0, 8.0],
        'Class': ['A', 'B', 'A', 'B']
    }
    return pd.DataFrame(data)


# =============================================================
# CLASE DE PRUEBAS UNITARIAS
# =============================================================
class TestExplorationUnitary:
    """Grupo de pruebas unitarias para el módulo exploration.py."""

    # -------------------------------------------------------------
    # TEST 1: visualizacion_inicial
    # -------------------------------------------------------------
    @patch('src.data.exploration.visualization')
    def test_visualizacion_inicial_log_messages(self, mock_vis, mock_data, caplog):
        """
        Verifica que la función visualizacion_inicial genera los logs de control
        y llama a los métodos clave de visualization.
        """
        df_org = mock_data.copy()
        
        # Configuramos mocks para evitar errores de ejecución
        mock_vis.size_df.return_value = mock_data.size
        mock_vis.cantidad_nulls.return_value = 1 

        with caplog.at_level(logging.DEBUG): 
            exploration.visualizacion_inicial(mock_data, df_org)
        
        # ------------------------------------
        # 1. Verificar logs de control
        # ------------------------------------
        assert "Inicia la impresión de la visualización inicial" in caplog.text
        assert "Termina la impresión de la visualización inicial" in caplog.text
        
        # ------------------------------------
        # 2. Verificar llamadas a funciones
        # ------------------------------------
        mock_vis.unique_class.assert_called_once()
        mock_vis.size_df.assert_called_once()


    # -------------------------------------------------------------
    # TEST 2: EDA
    # -------------------------------------------------------------
    @patch('src.data.exploration.visualization')
    def test_EDA_returns_cols_id_and_logs_info(self, mock_vis, mock_data, caplog):
        """
        Verifica que la función EDA retorna la lista de columnas ID correctamente 
        y registra los logs de control.
        """

        # ---------------------------------------------------------
        # CONFIGURACIÓN DE MOCKS NECESARIOS
        # ---------------------------------------------------------
        expected_ids = ['Feature1']
        mock_vis.detectar_columnas_id.return_value = expected_ids

        # Debe retornar tupla (numéricas, categóricas)
        mock_vis.variables_num_cat.return_value = (
            ['Feature1', 'Feature2'],  # numéricas
            ['Class']                  # categóricas
        )

        # resumen_dataset debe devolver un DataFrame para usar .head()
        mock_vis.resumen_dataset.return_value = pd.DataFrame({
            "col": [1, 2, 3]
        })

        # estadisticas_descriptivas debe retornar 2 DataFrames
        mock_vis.estadisticas_descriptivas.return_value = (
            pd.DataFrame({"a": [1, 2]}),
            pd.DataFrame({"b": ["x", "y"]})
        )

        # ---------------------------------------------------------
        # EJECUCIÓN DE LA PRUEBA
        # ---------------------------------------------------------
        with caplog.at_level(logging.INFO):
            result = exploration.EDA(mock_data)

        # ---------------------------------------------------------
        # VALIDACIONES
        # ---------------------------------------------------------

        # 1. Retorno correcto
        assert result == expected_ids
        
        # 2. Logs esperados
        assert "Inicia la impresión de los datos EDA" in caplog.text
        assert "Termina la impresión de los datos EDA" in caplog.text
        
        # 3. Verificar llamadas a métodos clave
        mock_vis.resumen_dataset.assert_called_once()
        mock_vis.estadisticas_descriptivas.assert_called_once()
