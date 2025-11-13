@echo off
echo [PASO 1] Creando el entorno virtual 'venv'...
python -m venv venv

echo.
echo [PASO 2] Instalando dependencias de requirements.txt...
call venv\Scripts\pip.exe install -r requirements.txt

echo.
echo =========================================================
echo  ¡CONFIGURACION COMPLETADA!
echo =========================================================
echo.
echo  Ahora, para empezar a trabajar, solo necesitas
echo  activar el entorno manualmente en tu terminal:
echo.
echo  Si estas en PowerShell: .\venv\Scripts\activate
echo  Si estas en CMD:         venv\Scripts\activate.bat
echo.