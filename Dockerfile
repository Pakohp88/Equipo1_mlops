FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

RUN git clone --branch integration https://github.com/Pakohp88/Equipo1_mlops.git

RUN mv Equipo1_mlops/* . && rm -rf Equipo1_mlops

RUN grep -v "pywin32" requirements.txt > requirements_linux.txt

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements_linux.txt

ENV PORT=8000

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la API con Uvicorn
CMD ["uvicorn", "src.api.app_equipo2:app", "--host", "0.0.0.0", "--port", "8000"]