# Usa una imagen base de Python
FROM python:3.13

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requisitos e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- DESCARGA DE MODELOS ---
# Crea la carpeta donde se guardarán los modelos
RUN mkdir -p /app/models

# Descarga de modelos .pth (reemplaza las URLs reales)
RUN wget -O /app/models/ensambleA_methodA_BBOX_output.pth "https://github.com/Fullops/Flask_invernadero/releases/download/1.0-alpha/ensambleA_methodA_BBOX_output.pth" && \
    wget -O /app/models/ensambleA_SoftVoting_3_modelos.pth "https://github.com/Fullops/Flask_invernadero/releases/download/1.0-alpha/ensambleA_SoftVoting_3_modelos.pth"

# Copia el resto de la aplicación al contenedor
COPY . .

# Expone el puerto en el que la aplicación Flask se ejecutará
EXPOSE 5000

# Define la variable de entorno para Flask
ENV FLASK_APP=app.py

# Ejecuta la aplicación Flask
CMD ["python", "app.py"]
