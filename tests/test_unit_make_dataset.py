# tests/test_unit_make_dataset.py

import logging
from click.testing import CliRunner
from src.data.make_dataset import main


def test_make_dataset_logs_info(tmp_path, caplog):
    """
    Verifica que el comando loguea el mensaje esperado.
    """
    # Crear archivo dummy de entrada
    input_file = tmp_path / "input.csv"
    input_file.write_text("some,data")

    # Ruta de salida dummy
    output_file = tmp_path / "output.csv"

    caplog.set_level(logging.INFO)

    runner = CliRunner()
    result = runner.invoke(main, [str(input_file), str(output_file)])

    assert result.exit_code == 0
    assert "making final data set from raw data" in caplog.text


def test_make_dataset_accepts_paths(tmp_path):
    """
    Verifica que el comando acepta paths válidos sin fallar.
    """
    input_file = tmp_path / "raw.csv"
    input_file.write_text("123")

    output_file = tmp_path / "processed.csv"

    runner = CliRunner()
    result = runner.invoke(main, [str(input_file), str(output_file)])

    assert result.exit_code == 0


def test_make_dataset_cli_execution(tmp_path, caplog):
    """
    Prueba la ejecución completa del CLI verificando logs, no stdout.
    """
    input_file = tmp_path / "entrada.csv"
    input_file.write_text("abc")

    output_file = tmp_path / "salida.csv"

    caplog.set_level(logging.INFO)

    runner = CliRunner()
    result = runner.invoke(main, [str(input_file), str(output_file)])

    assert result.exit_code == 0
    assert "making final data set from raw data" in caplog.text
