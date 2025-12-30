import streamlit as st
from sqlalchemy import create_engine, text
import sys

st.title("🕵️‍♂️ Diagnóstico de Conexión BD")

# 1. Verificar Librerías
st.write(f"Python Version: {sys.version}")
try:
    import pg8000
    st.success("✅ Librería pg8000 encontrada.")
except ImportError:
    st.error("❌ FALTA pg8000. Agregalo a requirements.txt")
    st.stop()

# 2. Leer Secreto
try:
    db_uri = st.secrets["DB_CONNECTION_STRING"]
    # Mostramos la URL enmascarada para verificar que no esté vacía o mal formada
    safe_uri = db_uri.split("@")[-1] if "@" in db_uri else "URI_INVALIDA"
    st.info(f"Intentando conectar a: ...@{safe_uri}")
except Exception as e:
    st.error(f"❌ No se pudo leer el secreto DB_CONNECTION_STRING: {e}")
    st.stop()

# 3. Intentar Conexión
if st.button("Probar Conexión Ahora"):
    try:
        # Creamos el motor. Importante: pg8000 a veces necesita ssl context explícito
        # pero probemos primero la cadena directa.
        engine = create_engine(db_uri)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            st.success("🎉 ¡CONEXIÓN EXITOSA!")
            st.write(f"Versión de la base de datos: {version}")
            
    except Exception as e:
        st.error("❌ FALLÓ LA CONEXIÓN")
        st.code(str(e), language="text") # Esto nos dará el error técnico exacto