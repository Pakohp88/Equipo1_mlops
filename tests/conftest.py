import sys, os
import pytest
import pandas as pd


# Ruta hasta la carpeta raíz del proyecto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# Ruta hasta la carpeta src
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)


@pytest.fixture
def tiny_dataset(tmp_path):
    """Genera un dataset mínimo de integración con 5 filas."""
    
    df = pd.DataFrame({
        "feature1": [1.0, 2.1, 1.5, 2.2, 1.8],
        "feature2": [3.2, 3.8, 3.5, 3.7, 3.4],
        "feature3": [5.1, 4.9, 5.0, 5.2, 5.3],
        "class":    [0, 1, 0, 1, 0]
    })
    
    file_path = tmp_path / "tiny_data.csv"
    df.to_csv(file_path, index=False)
    
    return file_path
