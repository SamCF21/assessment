# Arquitectura del Sistema - CROP WISE

## Arquitectura General del Sistema

CROP WISE utiliza una **Arquitectura de 5 Instancias Distribuidas** desplegada en nube privada del Tecnológico de Monterrey, con separación clara entre capas de cómputo, presentación y datos.

### Arquitectura Visual Principal

```
                     USUARIOS
                        │
            ┌───────────▼───────────┐
            │    LOAD BALANCER      │
            │    (Dockerizada)      │ ← 10.49.12.49:1010
            │    Round Robin        │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │   NUBE PRIVADA        │
            │   172.22.0.0/16       │
            │                       │
            │  ┌─────────────────┐  │
            │  │   FRONTEND      │  │
            │  │   (Instancia)   │  │ ← Next.js :3000
            │  │                 │  │
            │  │    ┌─────┐      │  │
            │  │    │ FE  │      │  │
            │  │    │     │      │  │
            │  │    └─────┘      │  │
            │  └─────────────────┘  │
            │           │           │
            │  ┌────────▼────────┐  │
            │  │   BACKEND       │  │
            │  │  (2 Instancias  │  │ ← Flask + Modelo ML
            │  │  Dockerizadas)  │  │   (Dockerizadas)
            │  │                 │  │
            │  │ ┌─────┐ ┌─────┐ │  │
            │  │ │BE #1│ │BE #2│ │  │ ← :5001 cada una
            │  │ │     │ │     │ │  │
            │  │ └─────┘ └─────┘ │  │
            │  └─────────────────┘  │
            │           │           │
            │      ┌────▼────┐      │
            │      │   BD    │      │ ← MySQL :3306
            │      │(Instancia)     │   (Independiente)
            │      │         │      │
            │      └─────────┘      │
            └───────────────────────┘
```
### Arquitectura de Alto Nivel

```
┌─────────────────────────────────────────────────────────────────┐
│                        USUARIO FINAL                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  LOAD BALANCER                                  │
│                  (Round Robin)                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
         ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Frontend   │ │   Frontend   │ │   Frontend   │
│   Instance   │ │   Instance   │ │   Instance   │
│  (Next.js)   │ │  (Next.js)   │ │  (Next.js)   │
│   Port 3000  │ │   Port 3001  │ │   Port 3002  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │ API Calls
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                    BACKEND LAYER                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Flask API RESTful                          │  │
│  │                                                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │   Auth      │  │    ML       │  │  Database   │      │  │
│  │  │  Service    │  │  Service    │  │   Service   │      │  │
│  │  │    JWT      │  │   PyTorch   │  │    MySQL    │      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────┬───────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                   DATA LAYER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │     MySQL       │  │   ML Models     │  │  Static Files │  │
│  │   Database      │  │   (.pkl files)  │  │   (Assets)    │  │
│  │                 │  │                 │  │               │  │
│  │ • Users         │  │ • Neural Net    │  │ • Images      │  │
│  │ • Crops         │  │   Architectures │  │ • Documents   │  │
│  │ • Predictions   │  │ • Trained       │  │ • Configs     │  │
│  │ • Climate Data  │  │   Weights       │  │               │  │
│  └─────────────────┘  └─────────────────┘  └───────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

## Distribución de las 5 Instancias

### 1. Load Balancer (Dockerizada) 
**Ubicación**: `10.49.12.49:1010` a través de la red del Tec.
- **Tecnología**: Nginx (Docker)
- **Algoritmo**: Round Robin
- **Función**: Punto de entrada único y distribución de carga
- **IP/Puerto**: `10.49.12.49:1010`
- **Estado**: Dockerizada

### 2. Frontend (Instancia Independiente)
**Ubicación**: Nube privada
- **Instancia**: `:3000`
- **Tecnología**: Next.js 
- **Función**: Interfaz de usuario
- **Estado**: No dockerizada

### 3-4. Backend (2 Instancias Dockerizadas)
**Ubicación**: Nube privada `172.22.x.x`
- **Instancia Backend #1**: `:5001` (Docker)
- **Instancia Backend #2**: `:5001` (Docker) 
- **Tecnología**: Flask + PyTorch
- **Función**: API REST y procesamiento ML
- **Estado**: Ambas dockerizadas

### 5. Base de Datos (Instancia Independiente)
**Ubicación**: Cómputo
- **Instancia**: `:3306`
- **Tecnología**: MySQL 8.0
- **Función**: Persistencia de datos
- **Estado**: No dockerizada

## Flujo de Arquitectura con 5 Instancias

### Distribución de Carga Real
```
Petición Usuario
       │
   ┌───▼────┐
   │   LB   │ ← Instancia 1 (Docker)
   │ Nginx  │
   └───┬────┘
       │
   Round Robin
    ┌──┴──┐
    ▼     ▼
┌─────┐ ┌─────┐ ← Instancias 3-4 (Docker)
│BE #1│ │BE #2│
│:5001│ │:5001│
└──┬──┘ └──┬──┘
   │       │
   └───┬───┘
       ▼
   ┌─────┐    ← Instancia 5 (No Docker)
   │ BD  │
   │:3306│
   └─────┘
```

## Configuración por Instancia

### Instancia 1: Load Balancer (Docker)
```yaml
Tipo: Dockerizada
Función: Nginx Load Balancer
CPU: Mínimo para routing
RAM: Buffer para conexiones
Puerto: 1010
Algoritmo: Round Robin
```

### Instancia 2: Frontend (Independiente)
```yaml
Tipo: No Dockerizada
Función: Next.js Static + SSR
CPU: Moderate para rendering
RAM: Cache para assets
Puerto: 3000
```

### Instancias 3-4: Backend (Docker x2)
```yaml
Tipo: Dockerizadas (idénticas)
Función: Flask API + ML Processing
CPU: Alto para cómputo ML
RAM: Suficiente para modelos PyTorch
Puerto: 5001 (cada una)
Modelo: Cargado en memoria
```

### Instancia 5: Base de Datos (Independiente)
```yaml
Tipo: No Dockerizada
Función: MySQL 8.0
CPU: Optimizado para queries
RAM: Buffer pools MySQL
Puerto: 3306
```

## Topología de Red
Para conectarnos en a la nube privada del Tec usamos una computadora (cómputo) que contiene nuestro backend y el modelo, ésta se conecta a la nube privada por medio de un switch conectado a un router que sirve de gateway para dar salida a ésta.

Del otro lado de la nube, la topología está puesta de la siguiente forma:
```
                ┌───────────────────────────┐
                │    Firewall/Router        │
                │         del Tec           │
                └─────────────┬─────────────┘
                              │
                ┌─────────────▼─────────────┐
                │            Red            │
                │       10.49.0.0/16        │
                └─────────────┬─────────────┘
                              │
                ┌─────────────▼───────────────┐
                │   Servidor Principal        │
                │     10.49.12.49             │
                │                             │
                │  ┌─────────────────────┐    │
                │  │  Load Balancer      │    │
                │  │   (Docker) :1010    │    │
                │  └──────────┬──────────┘    │
                │             │               │
                │     ┌───────┼───────┐       │
                │     ▼               ▼       │
                │  ┌──────┐        ┌──────┐   │
                │  │BE #1 │        │BE #2 │   │
                │  │(Dock)│        │(Dock)│   │
                │  │:5001 │        │:5001 │   │
                │  └──┬───┘        └───┬──┘   │
                │     └───────┬────────┘      │
                │             ▼               │
                │  ┌─────────────────────┐    │
                │  │   Frontend Next.js  │    │
                │  │  (Independiente)    │    │
                │  │       :3000         │    │
                │  └─────────────────────┘    │
                │             │               │
                │  ┌──────────▼──────────┐    │
                │  │   MySQL Database    │    │
                │  │  (Independiente)    │    │
                │  │       :3306         │    │
                │  └─────────────────────┘    │
                └─────────────────────────────┘
```

## Configuración de Instancias Real

| Instancia | Tipo | Tecnología | Puerto | Estado | Función |
|-----------|------|------------|--------|--------|---------|
| 1 | Load Balancer | Nginx | 1010 | Docker | Routing |
| 2 | Frontend | Next.js | 3000 | Independiente | UI |
| 3 | Backend #1 | Flask+ML | 5001 | Docker | API+ML |
| 4 | Backend #2 | Flask+ML | 5001 | Docker | API+ML |
| 5 | Database | MySQL | 3306 | Independiente | Datos |

---
