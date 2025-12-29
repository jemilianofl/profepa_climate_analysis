import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import utils 

# Configuración de página
st.set_page_config(page_title="Mapa Climatológico", page_icon="🗺️", layout="wide")
st.title("🗺️ Mapa Interactivo de Estaciones")

# Cargar Estaciones
df_estaciones = utils.cargar_estaciones()

if df_estaciones is None or df_estaciones.empty:
    st.error("No se pudo conectar a la base de datos.")
    st.stop()
    
# Selector de Estado
lista_estados = sorted(df_estaciones['ESTADO'].unique())

idx_default = 0
if 'estado_actual' in st.session_state:
    if st.session_state.estado_actual in lista_estados:
        idx_default = lista_estados.index(st.session_state.estado_actual)

estado_seleccionado = st.selectbox("Selecciona un Estado:", lista_estados, index=idx_default)
st.session_state.estado_actual = estado_seleccionado

# Filtrar
df_mapa = df_estaciones[df_estaciones['ESTADO'] == estado_seleccionado]

if df_mapa.empty:
    st.warning("No hay estaciones en este estado.")
else:
    # Mapa
    lat_center = df_mapa['LATITUD'].mean()
    lon_center = df_mapa['LONGITUD'].mean()
    
    m = folium.Map(location=[lat_center, lon_center], zoom_start=8)

    # Marcadores
    for _, row in df_mapa.iterrows():
        folium.CircleMarker(
            location=[row['LATITUD'], row['LONGITUD']],
            radius=5,
            color="#E63946",
            fill=True,
            fill_color="#E63946",
            fill_opacity=0.7,
            tooltip=f"<b>{row['NOMBRE']}</b>",
            popup=folium.Popup(f"ID: {row.get('ID', 'N/A')}<br>Estación: {row['NOMBRE']}", max_width=300)
        ).add_to(m)

    st_folium(m, height=500, width="100%") # Ancho responsive

st.info("👈 Navega al menú lateral para ver Estadísticas, Calidad de Datos y Predicciones.")