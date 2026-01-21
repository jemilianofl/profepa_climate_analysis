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
tab1, tab2, tab3 = st.tabs(["📊 Completitud", "📦 Outliers (Anomalías)", "📈 Normalidad"])

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
    
    # 1. Gráfico
    fig = px.box(df.sort_values("ESTACION"), x="ESTACION", y=variable, color="ESTACION")
    fig.update_layout(showlegend=False, xaxis_tickangle=-90, height=500)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.divider()
    st.subheader("📋 Reporte Detallado de Anomalías")
    st.markdown("Los siguientes datos se encuentran fuera del rango estadístico normal (Método IQR).")

    # 2. Cálculo Matemático de Outliers
    outliers_list = []
    
    # Iteramos por estación porque cada estación tiene su propio clima "normal"
    for nombre_estacion, grupo in df.groupby("ESTACION"):
        q1 = grupo[variable].quantile(0.25)
        q3 = grupo[variable].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        
        # Filtramos los datos anómalos
        anomalos = grupo[(grupo[variable] < limite_inferior) | (grupo[variable] > limite_superior)].copy()
        
        if not anomalos.empty:
            anomalos['Limite_Min_Teorico'] = round(limite_inferior, 2)
            anomalos['Limite_Max_Teorico'] = round(limite_superior, 2)
            outliers_list.append(anomalos)

    # 3. Mostrar y Descargar
    if outliers_list:
        df_outliers = pd.concat(outliers_list).sort_values(["ESTACION", "FECHA"])
        
        # Formatear fecha para que se vea limpia en el CSV/Tabla
        df_outliers['FECHA'] = df_outliers['FECHA'].dt.date 

        # Mostrar tabla interactiva
        st.dataframe(
            df_outliers[["ESTACION", "FECHA", variable, "Limite_Min_Teorico", "Limite_Max_Teorico"]], 
            use_container_width=True,
            hide_index=True
        )

        # Botón de Descarga
        csv = df_outliers.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Descargar CSV de Anomalías",
            data=csv,
            file_name=f"anomalias_{variable}_{estado}.csv",
            mime="text/csv",
        )
    else:
        st.success("✅ No se detectaron anomalías estadísticas (outliers) en los datos seleccionados.")

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