import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from sqlalchemy import create_engine, text
import numpy as np
import os
import sys
from dotenv import load_dotenv

st.set_page_config(page_title="Calidad de Datos", page_icon="🔍", layout="wide")
st.title("🔍 Auditoría de Calidad (Nivel Estatal)")

# Conexión
load_dotenv()
DB_URI = os.getenv("DB_CONNECTION_STRING")
if not DB_URI:
    try: DB_URI = st.secrets["DB_CONNECTION_STRING"]
    except: pass

if not DB_URI:
    st.error("❌ Faltan credenciales de BD.")
    st.stop()

engine = create_engine(DB_URI)

# Selectores
c1, c2 = st.columns(2)
with c1: estado = st.selectbox("Estado:", ["CAMPECHE", "YUCATAN", "QUINTANA ROO"])
with c2: variable = st.selectbox("Variable:", ["TMAX", "TMIN", "PRECIP"])

# Carga SQL
query = text(f"""
    SELECT l."FECHA", l."ESTACION", l."{variable}"
    FROM lecturas l
    JOIN estaciones e ON l."ESTACION" = e."NOMBRE"
    WHERE e."ESTADO" = :estado AND l."{variable}" IS NOT NULL
""")

with st.spinner("Analizando..."):
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"estado": estado})

if df.empty:
    st.warning("No hay datos.")
    st.stop()

df['FECHA'] = pd.to_datetime(df['FECHA'])

# Pestañas
tab1, tab2, tab3 = st.tabs(["📊 Completitud", "📦 Outliers", "📈 Normalidad"])

with tab1:
    st.markdown("### % de Datos Válidos por Estación")
    total_days = (df['FECHA'].max() - df['FECHA'].min()).days + 1
    counts = df.groupby('ESTACION').size().reset_index(name='Registros')
    counts['Completitud (%)'] = ((counts['Registros'] / total_days) * 100).clip(upper=100).round(1)
    
    st.dataframe(
        counts.sort_values('Completitud (%)').style.background_gradient(cmap="RdYlGn", subset=['Completitud (%)']),
        use_container_width=True
    )

with tab2:
    st.markdown("### Diagrama de Caja (Outliers)")
    fig = px.box(df.sort_values("ESTACION"), x="ESTACION", y=variable, color="ESTACION")
    fig.update_layout(showlegend=False, xaxis_tickangle=-90, height=600)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with tab3:
    st.markdown("### Q-Q Plot (Muestreo)")
    sample = df[variable].sample(n=min(len(df), 5000), random_state=42)
    qq = stats.probplot(sample, dist="norm")
    
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Datos'))
    
    # Línea roja
    x_line = np.array([min(qq[0][0]), max(qq[0][0])])
    y_line = qq[1][0] * x_line + qq[1][1]
    fig_qq.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', line=dict(color='red')))
    
    fig_qq.update_layout(title="Normalidad Global", xaxis_title="Teórico", yaxis_title="Real")
    st.plotly_chart(fig_qq, theme="streamlit", use_container_width=True)