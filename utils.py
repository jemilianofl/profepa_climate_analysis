import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import socket

# --- 1. PARCHE PARA STREAMLIT CLOUD (FORZAR IPv4) ---
# Sin esto, Streamlit Cloud falla al conectar con Supabase en muchos casos.
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    res = old_getaddrinfo(*args, **kwargs)
    return [r for r in res if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo
# ----------------------------------------------------

# Cargar variables de entorno (para local)
load_dotenv()

@st.cache_resource
def get_engine():
    """
    Crea la conexión a la base de datos una sola vez (Singleton).
    Maneja credenciales tanto locales (.env) como de Streamlit Cloud (secrets).
    """
    # 1. Intentar leer de variable de entorno (Local)
    db_uri = os.getenv("DB_CONNECTION_STRING")

    # 2. Si no existe, intentar leer de Streamlit Secrets (Nube)
    if not db_uri:
        try:
            db_uri = st.secrets["DB_CONNECTION_STRING"]
        except (FileNotFoundError, KeyError):
            pass

    # 3. Si sigue vacía, error
    if not db_uri:
        st.error("❌ Error de Configuración: No se encontró la variable 'DB_CONNECTION_STRING'. Revisa tus Secrets o tu archivo .env")
        return None

    # 4. Crear el motor con 'pool_pre_ping' para evitar desconexiones
    return create_engine(db_uri, pool_pre_ping=True)

@st.cache_data(ttl=3600)  # Guarda en caché por 1 hora
def cargar_estaciones():
    """Descarga el catálogo de estaciones."""
    engine = get_engine()
    if not engine: return pd.DataFrame()
    
    try:
        query = text('SELECT * FROM estaciones')
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error cargando estaciones: {e}")
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