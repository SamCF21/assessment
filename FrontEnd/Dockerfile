# Usa Node.js como base
FROM node:18

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todos los archivos al contenedor
COPY . .

# Instala dependencias
RUN npm install

# Compila el proyecto
RUN npm run build

# Expone el puerto que usará Next.js en producción
EXPOSE 3000

# Comando que se ejecuta al iniciar el contenedor
CMD ["npm", "start"]
