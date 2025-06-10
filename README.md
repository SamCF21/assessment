## Estructura General del Repositorio

### Introducción
Este repositorio contiene todos los archivos realizados como parte del Proyecto Final de Carrera (Assessment) para la carrera de Ingeniería en Tecnologías Computacionales del Tecnológico de Monterrey.

Este proyecto consiste en la creación de una página web conectada a un modelo de Inteligencia Artificial del cual se obtiene una recomendación de un cultivo a sembrar dependiendo de las condiciones climatológicas y de suelo dadas por el usuario.

El sistema cuenta con una arquitectura diseñada para poder ser instanciado en la nube privada con un balanceador de cargas. De manera local no se podrá probar esta funcionalidad, sin embargo, los archivos necesarios para esto están incluidos en el repositorio.

### Organización del Repositorio
#### [BackEnd](./Backend/)
Cuenta con los archivos necesarios para crear el modelo de IA, crear la base de datos y crear el servidor de Flask que conecta estos componentes.
#### [FrontEnd](./FrontEnd/)
Cuenta con los archivos necesarios para crear la instancia utilizando Next.js como principal framework, así como las rutas de conexión al backend.
#### [Docs](./Docs/)
Documentación centralizada del proyecto, cuenta con el esquema descrito en la sección de [Documentación Realizada](#documentación-realizada)
#### [conf.d](conf.d/) y archivos [cloudinit.yaml](cloudinit.yaml), [docker-compose.yml](/docker-compose.yml) y [nginx.conf](nginx.conf)
Archivos ocupados en la instanciación y configuración del proyecto en la nube privada.

### Documentación Realizada
- [README](README.md) con descripción del proyecto (este documento)
- [API](./Docs/Sistema.pdf)
- [Arquitectura](./Docs/Arquitectura.md)
- [Configuraciones](./Docs/Sistema.pdf)
- [Descripción de la aplicación de IA](./Docs/Sistema.pdf)
- [Diseño de base de datos y justificación](./Docs/Sistema.pdf)
- [Documentación del Sistema](./Docs/Sistema.pdf)
- [Manual de uso](./Docs/Manual.md)
- [Red](./Docs/Arquitectura.md)
- [Plan de conexión con el resto de la nube](./Docs/Arquitectura.md)
- [Pruebas realizadas](./Docs/Pruebas.pdf)

Las secciones de API, Configuraciones, Descripción de la aplicación de IA, Diseño de base de datos y justificación están contenidas en el archivo de [Documentación del Sistema](./Docs/Sistema.pdf), el cual está basado en el estándar IEEE 1016. Mientras que las secciones de Arquitectura, Red y Plan de conexión con el resto de la nube están contenidas en el documento de [Arquitectura](./Docs/Arquitectura.md).