import os
import glob
import re
import pandas as pd
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv
import socket
import zipfile

# --- PARCHE DE RED (Vital para Streamlit Cloud/Render) ---
try:
    old_getaddrinfo = socket.getaddrinfo
    def new_getaddrinfo(*args, **kwargs):
        res = old_getaddrinfo(*args, **kwargs)
        return [r for r in res if r[0] == socket.AF_INET]
    socket.getaddrinfo = new_getaddrinfo
except Exception:
    pass
# ---------------------------------------------------------

load_dotenv()

# --- CONFIGURACIÓN ---
DATA_FOLDER = "datos_climatologicos_diarios"
ZIP_FILE = "datos_peninsula.zip"
DB_URI = os.getenv("DB_CONNECTION_STRING") 
MAX_WORKERS = 4 
BATCH_SIZE = 20 # Reducimos el lote de archivos en memoria
CHUNK_SQL_SIZE = 500 # Reducimos el lote de inserción SQL (Vital para pg8000)

MAPA_ESTADOS = {
    "YUCATAN": "YUCATAN", "YUCATÁN": "YUCATAN", "YUC": "YUCATAN",
    "QUINTANA ROO": "QUINTANA ROO", "Q. ROO": "QUINTANA ROO", "Q ROO": "QUINTANA ROO",
    "CAMPECHE": "CAMPECHE", "CAMP": "CAMPECHE",
    "MOCK": "MOCK"
}

RE_LATITUD = re.compile(r"LATITUD\s*:\s*([\d\.-]+)")
RE_LONGITUD = re.compile(r"LONGITUD\s*:\s*([\d\.-]+)")
RE_ESTADO = re.compile(r"ESTADO\s*:\s*(.+)")
RE_NOMBRE = re.compile(r"NOMBRE\s*:\s*(.+)")

def conectar_db():
    if not DB_URI:
        raise ValueError("No se encontró DB_CONNECTION_STRING")
    # Forzar pg8000 si es necesario
    uri = DB_URI
    if uri.startswith("postgresql://"):
        uri = uri.replace("postgresql://", "postgresql+pg8000://", 1)
    
    return create_engine(uri, pool_pre_ping=True)

def normalizar_estado(texto):
    if not texto: return None
    texto = texto.split("LATITUD")[0].strip().upper()
    for k, v in MAPA_ESTADOS.items():
        if k in texto: return v
    return None

def procesar_txt(file_path):
    nombre_archivo = os.path.basename(file_path)
    
    try:
        encoding = 'utf-8'
        try:
            with open(file_path, 'r', encoding='utf-8') as f: f.read(100)
        except UnicodeDecodeError:
            encoding = 'latin-1'

        with open(file_path, 'r', encoding=encoding) as f:
            encabezado = [next(f) for _ in range(50)] 
            head_str = "".join(encabezado)
            
            # Metadatos
            est_match = RE_ESTADO.search(head_str)
            lat_match = RE_LATITUD.search(head_str)
            lon_match = RE_LONGITUD.search(head_str)
            nom_match = RE_NOMBRE.search(head_str)

            if not (est_match and lat_match and lon_match and nom_match): return None, None

            estado = normalizar_estado(est_match.group(1))
            if not estado: return None, None
            
            lon = float(lon_match.group(1))
            
            info = {
                "NOMBRE": nom_match.group(1).split("ESTADO")[0].strip(),
                "ESTADO": estado,
                "LATITUD": float(lat_match.group(1)),
                "LONGITUD": lon,
                "ALTITUD": 10.0, 
                "archivo": nombre_archivo
            }

            # Buscar Datos
            f.seek(0)
            start_line = 0
            found_data = False
            for i, line in enumerate(encabezado):
                if "FECHA" in line: 
                    start_line = i
                    found_data = True
                    break
            
            if not found_data: return None, None

            # Lectura Pandas
            try:
                df = pd.read_csv(
                    f, sep=r'\s+', skiprows=start_line + 2,
                    names=["FECHA", "PRECIP", "EVAP", "TMAX", "TMIN"],
                    encoding=encoding, engine='python', dtype=str # Leemos como string primero
                )
            except:
                return None, None

            # --- LIMPIEZA Y CONVERSIÓN RIGUROSA ---
            df["ESTACION"] = info["NOMBRE"]
            
            # Fechas
            df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce') 
            df = df.dropna(subset=['FECHA'])
            df = df[df['FECHA'].dt.year >= 2000]

            # Números (Vital para evitar el error '0.1')
            cols_num = ["PRECIP", "EVAP", "TMAX", "TMIN"]
            for col in cols_num:
                # Reemplazar 'NULO' por NaN y convertir a float
                df[col] = pd.to_numeric(df[col].replace('NULO', float('nan')), errors='coerce')

            if df.empty: return info, None 
            return info, df

    except StopIteration:
        return None, None
    except Exception:
        return None, None

def subir_lote(engine, metas, dfs):
    """Función de subida optimizada para pg8000."""
    if not metas: return
    
    try:
        with engine.begin() as conn:
            # 1. ESTACIONES
            df_metas = pd.DataFrame(metas)
            df_metas = df_metas.drop_duplicates(subset=['NOMBRE'])
            df_metas["ESTACION"] = df_metas["NOMBRE"] # Copiar columna NOMBRE a ESTACION
            
            cols_sql = ["ESTACION", "NOMBRE", "ESTADO", "LATITUD", "LONGITUD", "ALTITUD"]
            # Asegurar que existan todas las columnas
            for col in cols_sql:
                if col not in df_metas.columns: df_metas[col] = None
            
            df_metas = df_metas[cols_sql]
            
            # Subir Estaciones (usamos multi con chunksize pequeño)
            df_metas.to_sql("estaciones", conn, if_exists='append', index=False, method='multi', chunksize=CHUNK_SQL_SIZE)
            
            # 2. LECTURAS
            if dfs:
                df_lecturas = pd.concat(dfs)
                
                # Asegurar tipos float (Redundancia de seguridad)
                columnas_float = ["PRECIP", "EVAP", "TMAX", "TMIN"]
                for col in columnas_float:
                    df_lecturas[col] = df_lecturas[col].astype(float)

                # pg8000 a veces prefiere method=None (insert fila por fila) si multi falla,
                # pero probaremos method='multi' con chunksize pequeño (500) primero.
                print(f"   ⏳ Subiendo {len(df_lecturas)} lecturas...")
                df_lecturas.to_sql("lecturas", conn, if_exists='append', index=False, 
                                 chunksize=CHUNK_SQL_SIZE, method='multi')
                
    except Exception as e:
        print(f"❌ Error CRÍTICO subiendo lote a BD. Detalles:")
        print(e)
        # No detenemos el script, pero logueamos el error

def main():
    print("🚀 Iniciando Proceso ETL (Modo pg8000 Seguro)...")
    
    # 1. Descomprimir
    if not os.path.exists(DATA_FOLDER) or not os.listdir(DATA_FOLDER):
        if os.path.exists(ZIP_FILE):
            print(f"📦 Descomprimiendo {ZIP_FILE}...")
            with zipfile.ZipFile(ZIP_FILE, 'r') as z: z.extractall(DATA_FOLDER)
        else:
            print("⚠️ No hay datos.")

    # 2. Reset BD
    try:
        engine = conectar_db()
        print("✅ Conexión establecida.")
        print("🔨 Reiniciando tablas...")
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS lecturas CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS estaciones CASCADE;"))
            conn.execute(text("""
                CREATE TABLE estaciones (
                    "ESTACION" TEXT PRIMARY KEY, "NOMBRE" TEXT, "ESTADO" TEXT,
                    "LATITUD" FLOAT, "LONGITUD" FLOAT, "ALTITUD" FLOAT
                );
            """))
            conn.execute(text("""
                CREATE TABLE lecturas (
                    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    "FECHA" DATE, "ESTACION" TEXT REFERENCES estaciones("ESTACION"),
                    "PRECIP" FLOAT, "EVAP" FLOAT, "TMAX" FLOAT, "TMIN" FLOAT
                );
            """))
        print("✅ Tablas listas.")
    except Exception as e:
        print(f"❌ Error fatal en BD: {e}")
        return

    # 3. Archivos
    print("🔍 Buscando archivos...")
    todos = glob.glob(os.path.join(DATA_FOLDER, "**/*.txt"), recursive=True)
    keywords = ["camp", "yuc", "qroo", "quintana", "campeche"]
    files = [f for f in todos if any(kw in f.lower() for kw in keywords)]
    if not files and len(todos) > 0: files = todos

    print(f"📂 Procesando {len(files)} archivos...")
    
    # 4. Procesamiento
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
    
    print(f"📊 Finalizado: {count_ok} estaciones procesadas.")
    
    # 5. Índices
    print("🔧 Creando índices...")
    with engine.begin() as conn:
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_est ON estaciones ("ESTADO");'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_lec_fecha ON lecturas ("FECHA");'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_lec_est ON lecturas ("ESTACION");'))
        
    print("🏁 ¡ETL COMPLETADO EXITOSAMENTE!")

if __name__ == "__main__":
    main()