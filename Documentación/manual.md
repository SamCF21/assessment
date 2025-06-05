# Manual de uso
Hay 2 maneras principales para correr la aplicación localmente, haciendo uso del contenedor de Docker o instalando y configurando las aplicaciones y dependencias manualmente. Por facilidad de uso y efectividad de tiempo, recomendamos la instalación por medio de Docker, sin embargo, este manual también cubrirá los pasos detalladamente para correr la aplicación sin esta herramienta.

## Método 1. Creación via Docker
### Requisitos previos
#### 1. Instalar [Docker Desktop](https://www.docker.com/products/docker-desktop/) (incluye Docker Engine y Docker Compose).
#### 2. Clonar este repositorio de GitHub.
Desde la terminal, correr lo siguiente:
    ```bash
    git clone git@github.com:SamCF21/assessment.git
    cd assessment
    ```
#### 3. Iniciar la aplicación.
Desde la carpeta del proyecto, correr:
    ```bash
    docker-compose up
    ```
#### 4. Acceder a la aplicación en `https://localhost:5502`.

## Método 2. Instalación Manual
### Requisitos previos
#### 1. Instalación y configuración de MySQL.
Instalar [MySQL Community Server](https://dev.mysql.com/downloads/mysql/) y [MySQL Workbench](https://dev.mysql.com/downloads/workbench/).  
En MySQL Workbench, crear una conexión con la siguiente información:  
- Nombre: root
- Método de Conexión: Estándar (TCP/IP)
- Hostname: Predeterminado (127.0.0.1)
- Puerto: Predeterminado (3306)
- Username: root

Dentro de esta conexión, en el menú seleccionar 'Abrir Script SQL' y abrir el archivo [crop_classifier_db.sql](../crop_classifier_db.sql) y ejecutar el archivo completo para crear la base de datos.



#### 2. Instalación de Python y librerías.
Instalar [Python 3.13](https://www.python.org/downloads/).  
Instalar las librerías necesarias con el comando:
`pip install torch pandas scikit-learn matplotlib mysql-connector-python flask flask-cors PyJWT`
#### 3. Instalar [Node.js](https://nodejs.org/en/download)
#### 4. 

### Instalación de proyecto
#### 1. Hacer git pull del repositorio
#### 2. Crear la base de datos en la instancia de MySQL y usarla desde root (comando en terminal)
#### 3. Correr archivos de Python (Crop Recommendation y API)

#### 5. Preparación Backend
1. Ubicarse en la terminal en el proyecto, en el folder que contiene el archivo package.json
2. Correr el comando npm install para instalar todos los paquetes para correr la aplicación
3. Correr el comando npm start para incializar la aplicación (lo mismo que correr node server.js)
4. Desde la misma carpeta de backend, donde se encuentra el dockerfile, correr el comando: docker build -t myapp:1.0 . para crear la imagen
5. Correr el comando docker run -p 3001:3001 myapp:1.0
6. Para inicializar los múltiples contenedores, correr el comando docker-compose up --build -d (si se quita la opción -d se pueden ver los logs en la terminal de cuál instancia está respondiendo)
7. Generar un certificado auto-firmado para establecer conexión por HTTPS con el comando: openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout nginx-selfsigned.key -out nginx-selfsigned.crt