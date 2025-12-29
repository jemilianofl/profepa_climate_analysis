import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
DB_URI = os.getenv("DB_CONNECTION_STRING")

# Fallback para Streamlit Cloud
if not DB_URI:
    try:
        DB_URI = st.secrets["DB_CONNECTION_STRING"]
    except:
        pass

@st.cache_resource
def get_engine():
    """Crea la conexión a la base de datos una sola vez (Singleton)."""
    if not DB_URI:
        st.error("❌ Falta la cadena de conexión.")
        return None
    return create_engine(DB_URI)

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

    query = text("""
        SELECT l.* FROM lecturas l
        JOIN estaciones e ON l."ESTACION" = e."NOMBRE"
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