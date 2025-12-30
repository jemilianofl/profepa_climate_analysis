import os
import glob
import re
import pandas as pd
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from dotenv import load_dotenv
import socket

# --- PARCHE PARA GITHUB ACTIONS (FORZAR IPv4) ---
# GitHub Actions a veces intenta usar IPv6 para conectar a Supabase y falla.
# Esto obliga a usar IPv4.
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    res = old_getaddrinfo(*args, **kwargs)
    # Filtrar solo resultados familia AF_INET (IPv4)
    return [r for r in res if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo
# ------------------------------------------------

load_dotenv()

# --- CONFIGURACIÓN ---
DATA_FOLDER = "datos_climatologicos_diarios"
DB_URI = os.getenv("DB_CONNECTION_STRING") 
MAX_WORKERS = 4 
BATCH_SIZE = 50 

MAPA_ESTADOS = {
    "YUCATAN": "YUCATAN", "YUCATÁN": "YUCATAN", "YUC": "YUCATAN",
    "QUINTANA ROO": "QUINTANA ROO", "Q. ROO": "QUINTANA ROO", "Q ROO": "QUINTANA ROO",
    "CAMPECHE": "CAMPECHE", "CAMP": "CAMPECHE"
}

RE_LATITUD = re.compile(r"LATITUD\s*:\s*([\d\.-]+)")
RE_LONGITUD = re.compile(r"LONGITUD\s*:\s*([\d\.-]+)")
RE_ESTADO = re.compile(r"ESTADO\s*:\s*(.+)")
RE_NOMBRE = re.compile(r"NOMBRE\s*:\s*(.+)")

def conectar_db():
    if not DB_URI:
        # Imprimir parte del error para depurar si está vacío en Actions
        print("ERROR: La variable DB_CONNECTION_STRING está vacía.")
        raise ValueError("No se encontró DB_CONNECTION_STRING")
    return create_engine(DB_URI)

def normalizar_estado(texto):
    if not texto: return None
    texto = texto.upper().strip()
    for k, v in MAPA_ESTADOS.items():
        if k in texto: return v
    return None

def procesar_txt(file_path):
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            encabezado = [next(f) for _ in range(20)]
            head_str = "".join(encabezado)
            
            # 1. Metadatos
            est_match = RE_ESTADO.search(head_str)
            lat_match = RE_LATITUD.search(head_str)
            lon_match = RE_LONGITUD.search(head_str)
            nom_match = RE_NOMBRE.search(head_str)

            if not (est_match and lat_match and lon_match and nom_match): return None, None

            estado = normalizar_estado(est_match.group(1))
            lon = float(lon_match.group(1))
            
            # 2. FILTROS DE SEGURIDAD
            if not estado: return None, None
            # Filtro Geográfico: Si está al oeste de -94, NO es la península
            if lon < -94.0: return None, None 

            info = {
                "NOMBRE": nom_match.group(1).strip(),
                "ESTADO": estado,
                "LATITUD": float(lat_match.group(1)),
                "LONGITUD": lon,
                "archivo": os.path.basename(file_path)
            }

            # 3. Lectura de Datos
            f.seek(0)
            # Buscar línea de inicio
            for i, line in enumerate(encabezado):
                if line.strip().startswith("FECHA"):
                    start_line = i
                    break
            else: return None, None

            df = pd.read_csv(
                f, sep=r'\s+', skiprows=start_line + 2,
                names=["FECHA", "PRECIP", "EVAP", "TMAX", "TMIN"],
                encoding='latin-1', engine='python', dtype=str
            )

            # 4. Limpieza
            df["ESTACION"] = info["NOMBRE"]
            df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
            df = df.drop_duplicates(subset=['FECHA'], keep='last') # Anti-Crash SQL
            
            for c in ["PRECIP", "EVAP", "TMAX", "TMIN"]:
                df[c] = pd.to_numeric(df[c].replace("NULO", None), errors='coerce')
            
            df = df.dropna(subset=['FECHA'])
            df = df[df['FECHA'].dt.year >= 2000] # Solo datos recientes

            if df.empty: return info, None
            return info, df

    except Exception:
        return None, None

def subir_lote(engine, metas, dfs):
    if not metas: return
    try:
        with engine.begin() as conn:
            pd.DataFrame(metas).to_sql("estaciones", conn, if_exists='append', index=False, method='multi')
            pd.concat(dfs).to_sql("lecturas", conn, if_exists='append', index=False, chunksize=500, method='multi')
    except Exception as e:
        print(f"Error subiendo lote: {e}")

def main():
    print("🚀 Iniciando Actualización Automática...")
    
    # Intento de conexión con reintento simple
    try:
        engine = conectar_db()
        # Probar conexión simple antes de procesar
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Conexión a Base de Datos Exitosa (IPv4).")
    except Exception as e:
        print(f"❌ Error fatal conectando a la BD: {e}")
        return

    # IMPORTANTE: DROP para reiniciar tablas limpias cada vez
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS lecturas CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS estaciones CASCADE;"))
    except Exception as e:
        print(f"⚠️ Advertencia al reiniciar tablas: {e}")

    carpetas = ["CAMP", "YUC", "QROO"]
    files = []
    # Verifica que existan carpetas, si no, busca recursivo para evitar fallos si cambia la estructura
    for c in carpetas:
        path = os.path.join(DATA_FOLDER, c, "*.txt")
        encontrados = glob.glob(path)
        if not encontrados:
            print(f"⚠️ No se encontraron archivos en {path}, intentando búsqueda recursiva...")
            files.extend(glob.glob(os.path.join(DATA_FOLDER, "**/*.txt"), recursive=True))
            break
        files.extend(encontrados)
    
    # Eliminar duplicados de archivos si la búsqueda recursiva se activó
    files = list(set(files))
    
    print(f"📂 Procesando {len(files)} archivos...")
    
    batch_m, batch_d = [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exc:
        for info, df in tqdm(exc.map(procesar_txt, files), total=len(files)):
            if df is not None:
                batch_m.append(info)
                batch_d.append(df)
                if len(batch_m) >= BATCH_SIZE:
                    subir_lote(engine, batch_m, batch_d)
                    batch_m, batch_d = [], []
                    
    if batch_m: subir_lote(engine, batch_m, batch_d)
    
    print("🔧 Creando índices...")
    with engine.begin() as conn:
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_est ON estaciones ("ESTADO");'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_lec ON lecturas ("ESTACION", "FECHA");'))
        
    print("🏁 Actualización Completada.")

if __name__ == "__main__":
    main()