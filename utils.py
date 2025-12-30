import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import socket

# --- PARCHE IPv4 (Vital para pg8000 en Streamlit Cloud) ---
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    res = old_getaddrinfo(*args, **kwargs)
    return [r for r in res if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo
# ----------------------------------------------------------

load_dotenv()

@st.cache_resource
def get_engine():
    # 1. Obtener la URI
    db_uri = os.getenv("DB_CONNECTION_STRING")
    if not db_uri:
        try:
            db_uri = st.secrets["DB_CONNECTION_STRING"]
        except:
            pass
            
    if not db_uri:
        st.error("❌ Error: No se encontró la cadena de conexión.")
        return None

    # 2. Intentar crear el motor
    try:
        # Creamos el engine
        engine = create_engine(db_uri, pool_pre_ping=True)
        
        # 3. PRUEBA DE CONEXIÓN INMEDIATA
        # Esto forzará el error aquí mismo si no conecta
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        return engine
        
    except Exception as e:
        # ESTO TE MOSTRARÁ EL ERROR REAL EN PANTALLA
        st.error(f"❌ Error Fatal de Conexión: {e}")
        return None

@st.cache_data(ttl=3600)
def cargar_estaciones():
    engine = get_engine()
    if not engine: return pd.DataFrame()
    
    try:
        query = text('SELECT * FROM estaciones')
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error en la consulta SQL: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def cargar_lecturas_por_estado(estado):
    """Descarga lecturas filtradas por estado."""
    engine = get_engine()
    if not engine: return pd.DataFrame()

    # Nota: Asegúrate de que el JOIN sea correcto.
    # En tu ETL, la Primary Key de estaciones es "ESTACION", no "NOMBRE".
    # Lo cambié abajo para que sea más robusto (e."ESTACION"), pero si usas "NOMBRE" está bien.
    query = text("""
        SELECT l.* FROM lecturas l
        JOIN estaciones e ON l."ESTACION" = e."ESTACION"
        WHERE e."ESTADO" = :estado
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"estado": estado})
        return df
    except Exception as e:
        st.error(f"Error cargando lecturas: {e}")
        return pd.DataFrame()

def cargar_datos_procesados():
    """Función helper para cargar todo de golpe (opcional)."""
    estaciones = cargar_estaciones()
    return estaciones, None