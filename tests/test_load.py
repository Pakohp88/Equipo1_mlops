import pandas as pd
import pytest
from src.data.load import carga_datos

def test_carga_datos_retorna_dataframes(tmp_path):
    # Crear archivos CSV temporales
    mod_file = tmp_path / "modificado.csv"
    orig_file = tmp_path / "original.csv"

    mod_file.write_text("col1,col2\n1,2\n3,4")
    orig_file.write_text("colA,colB\n10,20\n30,40")

    df1, df2 = carga_datos(mod_file, orig_file)

    # Validar que retorna DataFrames
    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)

    # Validar contenido
    assert df1.shape == (2, 2)
    assert df2.shape == (2, 2)


def test_carga_datos_lectura_correcta(tmp_path):
    mod_file = tmp_path / "modificado.csv"
    orig_file = tmp_path / "original.csv"

    mod_file.write_text("a,b\n5,6")
    orig_file.write_text("x,y\n9,10")

    df1, df2 = carga_datos(mod_file, orig_file)

    assert list(df1.columns) == ["a", "b"]
    assert list(df2.columns) == ["x", "y"]
    assert df1.iloc[0].tolist() == [5, 6]
    assert df2.iloc[0].tolist() == [9, 10]


def test_carga_datos_archivo_no_existe(tmp_path):
    mod_file = tmp_path / "mod.csv"
    orig_file = tmp_path / "orig.csv"

    # No escribir archivos → deben fallar
    with pytest.raises(FileNotFoundError):
        carga_datos(mod_file, orig_file)
