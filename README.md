# 🌦️ Sistema de Análisis Climatológico y Predicción Estocástica (SACP)
### Península de Yucatán, México

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Supabase](https://img.shields.io/badge/Database-Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

---

## 📖 Descripción del Proyecto

Este repositorio aloja la plataforma integral desarrollada para la **Tesis de Maestría en Análisis de Datos Climáticos**. El sistema permite el monitoreo, auditoría de calidad y proyección estocástica de variables meteorológicas (Temperatura y Precipitación) en los estados de **Campeche, Yucatán y Quintana Roo**.

A diferencia de los tableros tradicionales, este sistema implementa una metodología de **"Caja Blanca"**, transparentando los cálculos matemáticos, el análisis de residuos y la cinemática del cambio climático.

---

## 📋 Tabla de Contenidos

1. [Arquitectura del Sistema](#-arquitectura-del-sistema)
2. [Características Científicas](#-características-científicas)
3. [Estructura del Repositorio](#-estructura-del-repositorio)
4. [Instalación y Despliegue](#-instalación-y-despliegue)
5. [Metodología de Predicción](#-metodología-de-predicción)
6. [Automatización (ETL)](#-automatización-etl)

---

## 🏗 Arquitectura del Sistema

El proyecto sigue una arquitectura desacoplada moderna:

* **Ingesta (ETL):** Scripts en Python que procesan archivos `.txt` crudos (formato CONAGUA/SMN), aplican filtros geográficos de seguridad y normalizan fechas.
* **Almacenamiento:** Base de datos relacional **PostgreSQL** alojada en la nube (Supabase).
* **Frontend:** Aplicación web interactiva construida con **Streamlit**.
* **Cómputo:** Librerías científicas (`SciPy`, `Pmdarima`, `Statsmodels`, `NumPy`).

---

## 🧪 Características Científicas

### 1. 🗺️ Mapa Interactivo de Estaciones
* Visualización geoespacial con `Folium`.
* Filtrado dinámico por entidad federativa.
* Optimización de renderizado para grandes volúmenes de puntos.

### 2. 🔍 Auditoría de Calidad de Datos
* **Completitud:** Tabla semafórica que calcula el % de datos válidos históricos por estación.
* **Detección de Outliers:** Diagramas de Caja (Boxplots) para identificar anomalías en sensores.
* **Prueba de Normalidad:** Gráficos Q-Q (Quantile-Quantile) con muestreo estadístico para validar la distribución gaussiana de los datos.

### 3. 📉 Cinemática Climática (Derivada)
Para evaluar la velocidad del cambio climático local, se calcula la primera derivada de la tendencia:
$$v(t) = \frac{dT_{trend}}{dt}$$
* Permite detectar periodos de **aceleración** (barras rojas) o desaceleración (barras azules) en el calentamiento.

### 4. 🔮 Predicciones "Caja Blanca" (SARIMA)
Se utiliza el algoritmo `auto_arima` para minimizar el criterio AIC (Akaike Information Criterion). El sistema expone:
* **Parámetros:** $(p,d,q) \times (P,D,Q)_{12}$ explicados en lenguaje natural.
* **Diagnóstico:** Histogramas y trazas de residuos para validar que el error sea "Ruido Blanco".
* **Proyección:** Intervalos de confianza al 95%.

---

## 📂 Estructura del Repositorio

mi-tesis-clima/
│
├── .github/workflows/      # CI/CD: Automatización de carga de datos
│   └── actualizar_datos.yml
│
├── datos_climatologicos_diarios/ # Datos crudos (Input)
│   ├── CAMP/
│   ├── YUC/
│   └── QROO/
│
├── pages/                  # Módulos de la aplicación
│   ├── Calidad_Datos.py    # Auditoría, Outliers y Q-Q Plots
│   ├── Estadisticas.py     # Climogramas y Tendencias lineales
│   └── Predicciones.py     # Modelo SARIMA y Cinemática
│
├── etl_supabase.py         # Script ETL (Extracción, Transformación y Carga)
├── Inicio.py               # Homepage (Mapa Interactivo)
├── utils.py                # Conexión a BD y Caché
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación

## 💻 Instalación y Despliegue

### Requisitos Previos
- Python 3.10 o superior.
- Cuenta en Supabase (PostgreSQL).

**Paso 1: Clonar**
```bash
git clone [https://github.com/TU_USUARIO/TU_REPO.git](https://github.com/TU_USUARIO/TU_REPO.git)
cd TU_REPO
```

**Paso 2: Entorno Virtual**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```
**Paso 3: Dependencias**
```bash
pip install -r requirements.txt
```

**Paso 4: Variables de Entorno**
Crea un archivo ```.env``` en la raíz (no lo subas a GitHub) con tu credencial:
```
DB_CONNECTION_STRING="postgresql://postgres:[TU_PASSWORD]@[TU_HOST]:5432/postgres"
```

**Paso 5: Ejecutar**

```bash
streamlit run Inicio.py
```

## 🤖 Automatización (ETL)
La base de datos se mantiene actualizada mediante GitHub Actions.

- Archivo: .github/workflows/actualizar_datos.yml

- Frecuencia: Semanal (Lunes 00:00 UTC).

- Proceso:

  1. Levanta un contenedor Ubuntu.

  2. Ejecuta etl_supabase.py.

  3. Aplica filtros de seguridad geográfica (elimina coordenadas erróneas fuera de la península).

  4. Sube los nuevos datos a Supabase.

**Autor:** José Emiliano Flores Pérez

Desarrollado como parte de mi proyecto de Tesis de Maestría junto a PROFEPA.