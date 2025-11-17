# Importar el código del archivo corregido: test_integration_training_testing_full.py

import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.model_selection import GridSearchCV

# Importación corregida asumiendo la estructura src/data
from src.data import training_testing 

# --- FUNCIÓN DE PRUEBA DE INTEGRACIÓN CORREGIDA ---

# FIX: Mockeamos fit_gridSearch en lugar de GridSearchCV.fit
@patch('src.data.training_testing.fit_gridSearch') 
@patch('src.data.training_testing.pd.read_csv')
@patch('src.data.training_testing.pd.DataFrame.to_csv')
def test_full_modeling_flow_integration_fixed(mock_to_csv, mock_read_csv, mock_fit_gridSearch):
    
    N_SAMPLES = 120 
    N_FEATURES = 10
    
    X_data = pd.DataFrame(np.random.rand(N_SAMPLES, N_FEATURES), 
                          columns=[f'feature_{i}' for i in range(N_FEATURES)])
    y_data = pd.Series(np.random.randint(0, 3, N_SAMPLES), name='class')
    
    # 1. Mockear carga de datos
    mock_read_csv.side_effect = [
        X_data, 
        y_data.to_frame()
    ]

    # 2. Mockear el resultado del estimador
    MockBestEstimator = MagicMock()
    MockBestEstimator.predict.return_value = np.zeros(N_SAMPLES)
    
    MockAdjustedGridSearch = MagicMock(spec=GridSearchCV)
    MockAdjustedGridSearch.best_estimator_ = MockBestEstimator
    
    mock_fit_gridSearch.side_effect = [MockAdjustedGridSearch] * 6

    # 3. Ejecutar el flujo completo de evaluación
    best_estimators = training_testing.evaluar_modelos()

    # 4. Asersiones
    assert isinstance(best_estimators, list)
    assert len(best_estimators) == 6
    assert mock_read_csv.call_count == 2
    
    # FIX: La asersión ahora se hace sobre la función que mockeamos
    assert mock_fit_gridSearch.call_count == 6
    
    for estimator in best_estimators:
        assert estimator == MockBestEstimator

