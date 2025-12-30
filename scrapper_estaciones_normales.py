# -*- coding: utf-8 -*-
"""
Script (v20) para descargar y renombrar archivos climatológicos.

VERSIÓN OPTIMIZADA, ROBUSTA Y CON LOGS.

- v20: Añade verificación de Content-Type para evitar "Falsos 404".
- v20: Añade un User-Agent de navegador para evitar bloqueos.
- v20: Añade reintento en códigos 404 (por si son fallos temporales).
- Utiliza ThreadPoolExecutor para descargas paralelas.
- Utiliza requests.Session para reutilización de conexiones y reintentos.
- Añade manejo del error SSL (SSLCertVerificationError).
- Añade medidor de tiempo y logging.
"""
import os
import re
import requests
import urllib3
import time
import logging
import random # <--- Nuevo para tiempos aleatorios
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURACIÓN CRÍTICA ---
# Bajamos de 10 a 2 para no saturar al servidor
MAX_WORKERS = 2 

LOG_EXITOS_FILE = "descargas_exitosas.log"
LOG_FALLOS_FILE = "descargas_fallidas.log"

# CORRECCIÓN IMPORTANTE: "CAMP" en lugar de "CAM"
ESTADOS_INTERES = ["CAMP", "YUC", "QROO"] 
# -----------------------------

def setup_logging():
    if os.path.exists(LOG_EXITOS_FILE): os.remove(LOG_EXITOS_FILE)
    if os.path.exists(LOG_FALLOS_FILE): os.remove(LOG_FALLOS_FILE)
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    log_exitos = logging.getLogger('exitos')
    log_exitos.setLevel(logging.INFO)
    log_exitos.propagate = False
    exitos_handler = logging.FileHandler(LOG_EXITOS_FILE, encoding='utf-8')
    exitos_handler.setFormatter(formatter)
    log_exitos.addHandler(exitos_handler)
    
    log_fallos = logging.getLogger('fallos')
    log_fallos.setLevel(logging.WARNING)
    log_fallos.propagate = False
    fallos_handler = logging.FileHandler(LOG_FALLOS_FILE, encoding='utf-8')
    fallos_handler.setFormatter(formatter)
    log_fallos.addHandler(fallos_handler)
    
    return log_exitos, log_fallos

def limpiar_nombre_archivo(nombre):
    return re.sub(r'[\\/*?:"<>|]', "_", nombre).strip()

def crear_session_robusta():
    """Crea una sesión con reintentos y User-Agent."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=8, # Aumentamos intentos
        backoff_factor=2, # Espera más tiempo entre fallos (2s, 4s, 8s...)
        status_forcelist=[404, 429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    session.headers.update(headers)
    
    session.verify = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session

def descargar_un_archivo(session, info_tarea, log_exitos, log_fallos):
    """Función de descarga individual con validación estricta y pausa."""
    url_absoluta_archivo, ruta_guardado, nuevo_nombre = info_tarea
    nombre_original = os.path.basename(url_absoluta_archivo)

    # --- PAUSA DE CORTESÍA ---
    # Esperamos entre 0.5 y 1.5 segundos antes de pedir el archivo
    # Esto evita que el servidor detecte un ataque.
    time.sleep(random.uniform(0.5, 1.5)) 
    # -------------------------

    try:
        # Nota: Puedes descomentar esto si quieres saltar archivos ya descargados
        # if os.path.exists(ruta_guardado): 
        #    print(f"     - ⏭️ Saltado (ya existe): {nuevo_nombre}")
        #    return

        contenido_archivo = session.get(url_absoluta_archivo, timeout=30) # Aumentamos timeout
        
        if contenido_archivo.status_code != 200:
            print(f"     - ❌ HTTP Error {contenido_archivo.status_code} para: {nombre_original}")
            log_fallos.warning(f"HTTP_ERROR_{contenido_archivo.status_code} | {nombre_original} | URL: {url_absoluta_archivo}")
            return

        content_type = contenido_archivo.headers.get('Content-Type', '').lower()
        if 'text/html' in content_type:
            print(f"     - ❌ Falso 404 (HTML) para: {nombre_original}")
            log_fallos.warning(f"FALSO_404_HTML | {nombre_original} | URL: {url_absoluta_archivo}")
            return

        with open(ruta_guardado, 'wb') as f:
            f.write(contenido_archivo.content)
        print(f"     - ✔️ Descargado: {nuevo_nombre}")
        log_exitos.info(f"{nuevo_nombre}")
            
    except requests.exceptions.RequestException as e:
        print(f"     - ❌ Error de red al descargar {nombre_original}: {e}")
        log_fallos.error(f"REQUEST_ERROR | {nombre_original} | URL: {url_absoluta_archivo} | Error: {e}")
    except Exception as e:
        print(f"     - ❌ Error inesperado en {nombre_original}: {e}")
        log_fallos.error(f"UNEXPECTED_ERROR | {nombre_original} | URL: {url_absoluta_archivo} | Error: {e}")

def descargar_archivos_diarios_final():
    inicio_proceso = time.perf_counter()
    log_exitos, log_fallos = setup_logging()

    base_url = "https://smn.conagua.gob.mx/"
    pagina_principal_url = urljoin(base_url, "es/climatologia/informacion-climatologica/normales-climatologicas-por-estado")
    base_url_archivos = "https://smn.conagua.gob.mx/tools/RESOURCES/Normales_Climatologicas/"

    print(f"-> Iniciando Scraper v22 (Modo Seguro: {ESTADOS_INTERES}) con {MAX_WORKERS} hilos... 🐢")
    
    carpeta_principal = "datos_climatologicos_diarios"
    os.makedirs(carpeta_principal, exist_ok=True)
    
    session = crear_session_robusta()

    try:
        print("\n-> Obteniendo la lista de estados...")
        response = session.get(pagina_principal_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        selector_estados = soup.find('select', id='listaestados')
        if not selector_estados:
            print("   Error: No se pudo encontrar el menú desplegable de estados.")
            return

        opciones_estados = [opt['value'] for opt in selector_estados.find_all('option') if opt.get('value')]
        codigos_todos = [opt.split('=')[-1] for opt in opciones_estados]
        
        # --- FILTRADO DE ESTADOS ---
        # Convertimos a mayúsculas para comparar
        codigos_filtrados = [c for c in codigos_todos if c.upper() in ESTADOS_INTERES]
        # ---------------------------

        print(f"   Se encontraron {len(codigos_todos)} estados en total.")
        print(f"   ⚠️ Filtrando solo: {codigos_filtrados}")

        for estado in codigos_filtrados:
            print(f"\n--- Procesando estado: {estado.upper()} ---")
            
            url_catalogo = f"{base_url_archivos}catalogo/cat_{estado}.html"
            carpeta_estado = os.path.join(carpeta_principal, estado.upper())
            os.makedirs(carpeta_estado, exist_ok=True)

            try:
                response_catalogo = session.get(url_catalogo)
                if response_catalogo.status_code != 200:
                    print(f"   No se pudo encontrar el catálogo para {estado.upper()}.")
                    continue

                soup_catalogo = BeautifulSoup(response_catalogo.content, 'html.parser')
                tabla = soup_catalogo.find('table')
                if not tabla:
                    print("   No se encontró la tabla en el catálogo.")
                    continue

                filas = tabla.find_all('tr')
                print(f"   Recolectando {len(filas)} estaciones para descargar...")

                tareas_para_descargar = []
                
                for fila in filas:
                    celdas = fila.find_all('td')
                    if len(celdas) < 5:
                        continue

                    nombre_estacion_raw = celdas[1].get_text(strip=True)
                    nombre_estacion_limpio = limpiar_nombre_archivo(nombre_estacion_raw)
                    enlace = celdas[4].find('a')
                    
                    if enlace and enlace.has_attr('href'):
                        href_relativo = enlace['href'].lstrip('../')
                        url_absoluta_archivo = urljoin(base_url_archivos, href_relativo)
                        nombre_original = os.path.basename(url_absoluta_archivo)
                        nuevo_nombre = f"{nombre_estacion_limpio}_{nombre_original}"
                        ruta_guardado = os.path.join(carpeta_estado, nuevo_nombre)
                        
                        tareas_para_descargar.append( (url_absoluta_archivo, ruta_guardado, nuevo_nombre) )
                
                if not tareas_para_descargar:
                    print("   No hay archivos para descargar.")
                    continue
                    
                print(f"   Descargando {len(tareas_para_descargar)} archivos con {MAX_WORKERS} hilos...")
                
                worker_preparado = partial(descargar_un_archivo, 
                                           session, 
                                           log_exitos=log_exitos, 
                                           log_fallos=log_fallos)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    list(executor.map(worker_preparado, tareas_para_descargar))
                
                print(f"   ✅ Fin de descarga para: {estado.upper()}")

            except Exception as e:
                print(f"   Error procesando {estado.upper()}: {e}")
                log_fallos.error(f"ERROR_ESTADO | {estado.upper()} | {e}")

    except Exception as e:
        print(f"\nOcurrió un error general: {e}")
        log_fallos.critical(f"ERROR_GENERAL | {e}")

    fin_proceso = time.perf_counter()
    duracion_total = fin_proceso - inicio_proceso
    print("\n" + "="*30)
    print(f" 🏁 PROCESO TERMINADO EN {duracion_total/60:.2f} min")
    print("="*30)

if __name__ == "__main__":
    descargar_archivos_diarios_final()