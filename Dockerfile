# Usa una imagen base de Python
FROM python:3.13

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requisitos e instala las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación al directorio de trabajo
COPY . .

# Expone el puerto en el que la aplicación Flask se ejecutará
EXPOSE 5000

# Define la variable de entorno para Flask
ENV FLASK_APP=app.py

# Ejecuta la aplicación Flask
CMD ["python", "app.py"]
