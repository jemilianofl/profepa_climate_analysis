import os
import glob
import re
import pandas as pd
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from dotenv import load_dotenv

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
            # Filtro Geográfico: Si está al oeste de -94, NO es la península (es Michoacán/Jalisco infiltrado)
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
    with engine.begin() as conn:
        pd.DataFrame(metas).to_sql("estaciones", conn, if_exists='append', index=False, method='multi')
        pd.concat(dfs).to_sql("lecturas", conn, if_exists='append', index=False, chunksize=500, method='multi')

def main():
    print("🚀 Iniciando Actualización Automática...")
    engine = conectar_db()
    
    # IMPORTANTE: DROP para reiniciar tablas limpias cada vez
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS lecturas CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS estaciones CASCADE;"))

    carpetas = ["CAMP", "YUC", "QROO"]
    files = []
    for c in carpetas:
        files.extend(glob.glob(os.path.join(DATA_FOLDER, c, "*.txt")))
    
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
    
    # Índices finales
    with engine.begin() as conn:
        conn.execute(text('CREATE INDEX idx_est ON estaciones ("ESTADO");'))
        conn.execute(text('CREATE INDEX idx_lec ON lecturas ("ESTACION", "FECHA");'))
        
    print("🏁 Actualización Completada.")

if __name__ == "__main__":
    main()