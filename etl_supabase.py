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
import zipfile # <--- NUEVO

# --- PARCHE PARA GITHUB ACTIONS (FORZAR IPv4) ---
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    res = old_getaddrinfo(*args, **kwargs)
    return [r for r in res if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo
# ------------------------------------------------

load_dotenv()

# --- CONFIGURACIÓN ---
DATA_FOLDER = "datos_climatologicos_diarios"
ZIP_FILE = "datos_peninsula.zip" # <--- NUEVO
DB_URI = os.getenv("DB_CONNECTION_STRING") 
MAX_WORKERS = 4 
BATCH_SIZE = 50 

# --- CORRECCIÓN 1: AGREGAMOS "MOCK" PARA QUE ACEPTE TUS DATOS FALSOS ---
MAPA_ESTADOS = {
    "YUCATAN": "YUCATAN", "YUCATÁN": "YUCATAN", "YUC": "YUCATAN",
    "QUINTANA ROO": "QUINTANA ROO", "Q. ROO": "QUINTANA ROO", "Q ROO": "QUINTANA ROO",
    "CAMPECHE": "CAMPECHE", "CAMP": "CAMPECHE",
    "MOCK": "MOCK" # <--- ¡IMPORTANTE!
}

RE_LATITUD = re.compile(r"LATITUD\s*:\s*([\d\.-]+)")
RE_LONGITUD = re.compile(r"LONGITUD\s*:\s*([\d\.-]+)")
RE_ESTADO = re.compile(r"ESTADO\s*:\s*(.+)")
RE_NOMBRE = re.compile(r"NOMBRE\s*:\s*(.+)")

def conectar_db():
    if not DB_URI:
        print("ERROR: La variable DB_CONNECTION_STRING está vacía.")
        raise ValueError("No se encontró DB_CONNECTION_STRING")
    return create_engine(DB_URI)

def normalizar_estado(texto):
    if not texto: return None
    # Truco: Si el regex capturó algo largo como "MOCK   LATITUD...", cortamos en el primer espacio
    texto = texto.split()[0].upper().strip() 
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

            if not (est_match and lat_match and lon_match and nom_match): 
                # print(f"⚠️ Rechazado {os.path.basename(file_path)}: Falta metadata en header")
                return None, None

            estado_raw = est_match.group(1)
            estado = normalizar_estado(estado_raw)
            lon = float(lon_match.group(1))
            
            # 2. FILTROS DE SEGURIDAD
            if not estado: 
                # print(f"⚠️ Rechazado {os.path.basename(file_path)}: Estado '{estado_raw}' no reconocido")
                return None, None
            
            # Filtro Geográfico (Opcional: Comenta esto si tus datos Mock tienen coordenadas 0.0)
            # if lon < -94.0: return None, None 

            info = {
                # Limpieza extra en el nombre por si el regex agarró basura
                "NOMBRE": nom_match.group(1).split("ESTADO")[0].strip(),
                "ESTADO": estado,
                "LATITUD": float(lat_match.group(1)),
                "LONGITUD": lon,
                "ALTITUD": 10.0, # Valor por defecto si no lo leemos
                "archivo": os.path.basename(file_path)
            }

            # 3. Lectura de Datos
            f.seek(0)
            start_line = 0
            found_data = False
            for i, line in enumerate(encabezado):
                if "FECHA" in line and "PRECIP" in line: # Búsqueda más flexible
                    start_line = i
                    found_data = True
                    break
            
            if not found_data: 
                return None, None

            df = pd.read_csv(
                f, sep=r'\s+', skiprows=start_line + 2,
                names=["FECHA", "PRECIP", "EVAP", "TMAX", "TMIN"],
                encoding='latin-1', engine='python', dtype=str
            )

            # 4. Limpieza
            df["ESTACION"] = info["NOMBRE"] # Enlace clave con la tabla padre
            df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
            df = df.drop_duplicates(subset=['FECHA'], keep='last')
            
            for c in ["PRECIP", "EVAP", "TMAX", "TMIN"]:
                df[c] = pd.to_numeric(df[c].replace("NULO", None), errors='coerce')
            
            df = df.dropna(subset=['FECHA'])
            df = df[df['FECHA'].dt.year >= 2000]

            if df.empty: 
                return info, None # Retornamos info aunque no haya lecturas, para poblar estaciones
            
            return info, df

    except Exception as e:
        # print(f"Error procesando {file_path}: {e}")
        return None, None

def subir_lote(engine, metas, dfs):
    if not metas: return
    try:
        with engine.begin() as conn:
            # Subir estaciones (usando ignorar duplicados si ya existen es complejo en pandas puro, 
            # pero como borramos la tabla al inicio, el append está bien)
            df_metas = pd.DataFrame(metas).drop_duplicates(subset=['NOMBRE'])
            # Renombramos columna NOMBRE -> ESTACION para que coincida con la Primary Key SQL
            df_metas = df_metas.rename(columns={"NOMBRE": "ESTACION"})
            
            # Ajuste de columnas para coincidir con SQL
            df_metas = df_metas[["ESTACION", "NOMBRE", "ESTADO", "LATITUD", "LONGITUD", "ALTITUD"]]
            
            df_metas.to_sql("estaciones", conn, if_exists='append', index=False, method='multi')
            
            if dfs:
                pd.concat(dfs).to_sql("lecturas", conn, if_exists='append', index=False, chunksize=1000, method='multi')
    except Exception as e:
        print(f"❌ Error subiendo lote: {e}")

def main():
    print("🚀 Iniciando Actualización Automática...")
    
    # 1. Descomprimir si es necesario
    if not os.path.exists(DATA_FOLDER) or not os.listdir(DATA_FOLDER):
        if os.path.exists(ZIP_FILE):
            print(f"📦 Descomprimiendo {ZIP_FILE}...")
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(DATA_FOLDER)
        else:
            print("⚠️ No se encontraron datos ni ZIP.")

    try:
        engine = conectar_db()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Conexión a Base de Datos Exitosa.")
    except Exception as e:
        print(f"❌ Error fatal conectando a la BD: {e}")
        return

    # --- CORRECCIÓN 2: CREACIÓN EXPLÍCITA DE TABLAS ---
    print("🔨 Reiniciando esquema de tablas...")
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS lecturas CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS estaciones CASCADE;"))
            
            # Crear Estaciones
            conn.execute(text("""
                CREATE TABLE estaciones (
                    "ESTACION" TEXT PRIMARY KEY,
                    "NOMBRE" TEXT,
                    "ESTADO" TEXT,
                    "LATITUD" FLOAT,
                    "LONGITUD" FLOAT,
                    "ALTITUD" FLOAT
                );
            """))
            # Crear Lecturas
            conn.execute(text("""
                CREATE TABLE lecturas (
                    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    "FECHA" DATE,
                    "ESTACION" TEXT REFERENCES estaciones("ESTACION"),
                    "PRECIP" FLOAT,
                    "EVAP" FLOAT,
                    "TMAX" FLOAT,
                    "TMIN" FLOAT
                );
            """))
        print("✅ Tablas creadas correctamente.")
    except Exception as e:
        print(f"❌ Error creando tablas: {e}")
        return
    # --------------------------------------------------

    carpetas = ["CAMP", "YUC", "QROO"]
    files = []
    # Búsqueda robusta (recursiva)
    files = glob.glob(os.path.join(DATA_FOLDER, "**/*.txt"), recursive=True)
    
    print(f"📂 Procesando {len(files)} archivos...")
    
    batch_m, batch_d = [], []
    count_ok = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exc:
        for info, df in tqdm(exc.map(procesar_txt, files), total=len(files)):
            if info is not None:
                batch_m.append(info)
                if df is not None:
                    batch_d.append(df)
                    count_ok += 1
                
                if len(batch_m) >= BATCH_SIZE:
                    subir_lote(engine, batch_m, batch_d)
                    batch_m, batch_d = [], []
                    
    if batch_m: subir_lote(engine, batch_m, batch_d)
    
    print(f"📊 Se procesaron con éxito {count_ok} archivos con lecturas.")
    
    print("🔧 Creando índices...")
    with engine.begin() as conn:
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_est ON estaciones ("ESTADO");'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_lec_fecha ON lecturas ("FECHA");'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_lec_est ON lecturas ("ESTACION");'))
        
    print("🏁 Actualización Completada.")

if __name__ == "__main__":
    main()