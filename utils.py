import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
import socket

# --- PARCHE IPv4 (Vital para pg8000 en Streamlit Cloud) ---
try:
    old_getaddrinfo = socket.getaddrinfo
    def new_getaddrinfo(*args, **kwargs):
        res = old_getaddrinfo(*args, **kwargs)
        return [r for r in res if r[0] == socket.AF_INET]
    socket.getaddrinfo = new_getaddrinfo
except Exception:
    pass
# ----------------------------------------------------------

@st.cache_resource
def get_engine():
    """
    Crea la conexión a la base de datos usando pg8000.
    """
    # 1. Obtener la URI (Prioridad: Secrets > Environment)
    try:
        db_uri = st.secrets["DB_CONNECTION_STRING"]
    except:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            db_uri = os.getenv("DB_CONNECTION_STRING")
        except:
            db_uri = None

    if not db_uri:
        st.error("❌ Error: No se encontró 'DB_CONNECTION_STRING'.")
        return None

    # 2. FORZAR DRIVER pg8000 (El truco mágico)
    # Si la cadena viene como 'postgresql://', la cambiamos a 'postgresql+pg8000://'
    # Esto asegura que Streamlit Cloud no use drivers binarios rotos.
    if db_uri.startswith("postgresql://"):
        db_uri = db_uri.replace("postgresql://", "postgresql+pg8000://", 1)

    # 3. Crear motor
    try:
        engine = create_engine(db_uri, pool_pre_ping=True)
        return engine
    except Exception as e:
        st.error(f"❌ Error creando el motor de base de datos: {e}")
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
        st.error(f"Error cargando estaciones: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def cargar_lecturas_por_estado(estado):
    engine = get_engine()
    if not engine: return pd.DataFrame()

    try:
        query = text("""
            SELECT l.* FROM lecturas l
            JOIN estaciones e ON l."ESTACION" = e."ESTACION"
            WHERE e."ESTADO" = :estado
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"estado": estado})
        return df
    except Exception as e:
        st.error(f"Error cargando lecturas: {e}")
        return pd.DataFrame()