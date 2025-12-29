import streamlit as st
import pandas as pd
import altair as alt
import sys
import os

# Importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

st.set_page_config(page_title="Climograma", page_icon="🌤️", layout="wide")
st.title("🌤️ Climograma y Tendencias")

# Carga de datos
df_estaciones = utils.cargar_estaciones()
if df_estaciones is None or df_estaciones.empty:
    st.error("Error de conexión.")
    st.stop()

# Selector
lista_estados = sorted(df_estaciones['ESTADO'].unique())
estado_seleccionado = st.selectbox("Selecciona Estado:", lista_estados)

# Cargar lecturas
with st.spinner(f"Cargando datos de {estado_seleccionado}..."):
    df_lecturas = utils.cargar_lecturas_por_estado(estado_seleccionado)

if df_lecturas.empty:
    st.warning("No hay datos.")
    st.stop()

# Procesamiento
df_lecturas['FECHA'] = pd.to_datetime(df_lecturas['FECHA'])
for col in ['TMAX', 'TMIN', 'PRECIP']:
    df_lecturas[col] = pd.to_numeric(df_lecturas[col], errors='coerce')

# Agrupación Mensual
df_lecturas['Year'] = df_lecturas['FECHA'].dt.year
df_lecturas['Month'] = df_lecturas['FECHA'].dt.month

mensual_historico = df_lecturas.groupby(['Year', 'Month']).agg({
    'PRECIP': 'sum', 'TMAX': 'mean', 'TMIN': 'mean'
}).reset_index()

mensual_historico['TMED'] = (mensual_historico['TMAX'] + mensual_historico['TMIN']) / 2
climograma_data = mensual_historico.groupby('Month').median().reset_index()

nombres_meses = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 
                 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
climograma_data['NombreMes'] = climograma_data['Month'].map(nombres_meses)

# --- VISUALIZACIÓN ---
st.subheader(f"Climograma: {estado_seleccionado}")

base = alt.Chart(climograma_data).encode(
    x=alt.X('NombreMes:N', sort=list(nombres_meses.values()), title=None, scale=alt.Scale(paddingInner=0.05))
)

barras = base.mark_bar(color='#1f77b4', opacity=0.9).encode(
    y=alt.Y('PRECIP:Q', axis=alt.Axis(title='Precipitación (mm)', titleColor='#1f77b4', orient='right'))
)

linea = base.mark_line(color='#d62728', strokeWidth=3).encode(
    y=alt.Y('TMED:Q', scale=alt.Scale(zero=False, padding=10), axis=alt.Axis(title='Temperatura (°C)', titleColor='#d62728', orient='left'))
)

puntos = base.mark_circle(color='#d62728', size=60).encode(y='TMED:Q', tooltip=['NombreMes', 'TMED', 'PRECIP'])

st.altair_chart((barras + linea + puntos).resolve_scale(y='independent').properties(height=400), theme="streamlit", use_container_width=True)

# Tabla
df_tabla = climograma_data[['NombreMes', 'PRECIP', 'TMED']].set_index('NombreMes').T
st.caption("Tabla de Medianas Históricas")
st.table(df_tabla.style.format("{:.1f}"))

# Tendencia
st.divider()
st.subheader("📉 Tendencia Histórica")
historico_grafica = mensual_historico.copy()
historico_grafica['FechaGrafica'] = pd.to_datetime(historico_grafica[['Year', 'Month']].assign(DAY=1))

chart = alt.Chart(historico_grafica).mark_line(color='gray').encode(
    x='FechaGrafica:T', y='TMED:Q', tooltip=['FechaGrafica', 'TMED']
).properties(height=300)

trend = chart.transform_regression('FechaGrafica', 'TMED').mark_line(color='red', strokeDash=[5,5])

st.altair_chart(chart + trend, theme="streamlit", use_container_width=True)