import pandas as pd
import pytest

# --------------------------
# Funciones utilitarias
# --------------------------
def size_df(df):
    """Devuelve el número total de celdas del dataframe."""
    return df.shape[0] * df.shape[1]

def count_null(df):
    """Devuelve el total de valores nulos en el dataframe."""
    return int(df.isnull().sum().sum())

# --------------------------
# Función visualizacion_inicial simplificada
# --------------------------
def visualizacion_inicial(df, df_org):
    # Conteo de valores nulos
    total_celdas = size_df(df)
    total_nulls = count_null(df)

    print(f"Total de celdas: {total_celdas}")
    print(f"Total de valores nulos: {total_nulls} ({((total_nulls / total_celdas) * 100):.2f}%)")

    filas_completas = df.dropna().shape[0]
    print(f"Filas completamente válidas (sin NaN): {filas_completas}")

    diff = df.shape[0] - filas_completas
    print(f"Diferencia (filas incompletas): {diff}")

# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def df_muestra():
    return pd.DataFrame({
        "A": [1, 2, 3, None],
        "B": ["x", "y", "z", "w"],
        "class": ["ok", "ok", "fail", None]
    })

@pytest.fixture
def df_org(df_muestra):
    return df_muestra.copy()

# --------------------------
# Test
# --------------------------
def test_visualizacion_inicial(df_muestra, df_org):
    visualizacion_inicial(df_muestra, df_org)
    # Validaciones simples
    assert size_df(df_muestra) == 12
    assert count_null(df_muestra) == 2
