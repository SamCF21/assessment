# Manual de uso
Debido a la complejidad de la aplicación se requieren de varios componentes para ejecutarla debidamente. Estos se cubren de manera puntual en Requisitos Previos y son indispensables para el correcto funcionamiento del sistema.

### Requisitos previos
#### 1. Hacer git pull del repositorio
Desde la terminal, correr el comando `git clone git@github.com:SamCF21/assessment.git`.  
Esto si se desea clonar por SSH, aunque es de libre elección.

#### 2. Instalación y configuración de MySQL.
Instalar [MySQL Community Server](https://dev.mysql.com/downloads/mysql/) y [MySQL Workbench](https://dev.mysql.com/downloads/workbench/).  
En MySQL Workbench, crear una conexión con la siguiente información:  
- Nombre: root
- Método de Conexión: Estándar (TCP/IP)
- Hostname: Predeterminado (127.0.0.1)
- Puerto: Predeterminado (3306)
- Username: root

Dentro de esta conexión, en el menú seleccionar 'Abrir Script SQL', abrir el archivo [crop_classifier_db.sql](../Backend/crop_classifier_db.sql) y ejecutar el archivo completo para crear la base de datos.

#### 3. Instalación de Python y librerías.
Instalar [Python 3.13](https://www.python.org/downloads/).  
Instalar las librerías necesarias con el comando `pip install -r requirements.txt`.

#### 4. Instalación de Node.js  y sus dependencias.
Instalar [Node.js](https://nodejs.org/en/download).  
Instalar las dependencias necesarias con el comando `npm install`.

### Ejecución del proyecto
#### 1. Crear BackEnd
Desde la carpeta de BackEnd, correr los comandos:  
`python3 2_6croprecommendation.py`  
`python3 load_trained_weights.py`  
`python3 api_mysql.py`  

#### 2. Crear FrontEnd
Desde la carpeta de FrontEnd, correr el comando:  
`npm run dev` 

#### 3. Ejecutar aplicación
La aplicación se iniciará automáticamente y será accesible desde [http://localhost:3000](http://localhost:3000).
