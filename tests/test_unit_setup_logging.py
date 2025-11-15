import logging
import sys
import os

# IMPORT CORRECTO PARA TU ESTRUCTURA
from src.data.logging_config import setup_logging


def test_setup_logging_creates_handlers(tmp_path):
    """
    Verifica que setup_logging:
    - Limpia handlers previos
    - Crea handler de consola y archivo
    - Establece niveles correctamente
    """

    log_file = tmp_path / "test_log.log"

    setup_logging(
        log_file=str(log_file),
        file_level=logging.WARNING,
        console_level=logging.DEBUG
    )

    root_logger = logging.getLogger()

    # Debe haber exactamente 2 handlers (consola y archivo)
    assert len(root_logger.handlers) == 2

    console_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert len(console_handlers) == 1

    file_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == 1

    assert console_handlers[0].level == logging.DEBUG
    assert file_handlers[0].level == logging.WARNING


def test_setup_logging_writes_to_file(tmp_path):
    """
    Verifica que se escriban logs en archivo.
    """

    log_file = tmp_path / "log_output.log"

    setup_logging(
        log_file=str(log_file),
        file_level=logging.INFO,
        console_level=logging.DEBUG
    )

    logger = logging.getLogger()
    logger.info("MENSAJE_PRUEBA")

    with open(log_file, "r") as file:
        contenido = file.read()

    assert "MENSAJE_PRUEBA" in contenido


def test_setup_logging_clears_previous_handlers(tmp_path):
    """
    Verifica que los handlers previos se eliminen correctamente.
    """

    log_file = tmp_path / "clear_test.log"

    root_logger = logging.getLogger()

    fake_handler = logging.StreamHandler(sys.stdout)
    root_logger.addHandler(fake_handler)

    assert len(root_logger.handlers) > 0  # Antes de llamar setup_logging

    setup_logging(log_file=str(log_file))

    # Ahora deben existir solo los nuevos handlers
    assert len(root_logger.handlers) == 2
    assert fake_handler not in root_logger.handlers
