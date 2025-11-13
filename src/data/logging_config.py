import logging
import sys

def setup_logging(log_file="Eq1MLOps.log", file_level=logging.INFO, console_level=logging.DEBUG):
    root_logger = logging.getLogger()

    # Limpia handlers previos
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Nivel mínimo general
    root_logger.setLevel(logging.DEBUG)

    # ------------------------------
    # Handler para consola
    # ------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ------------------------------
    # Handler para archivo
    # ------------------------------
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
