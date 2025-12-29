import streamlit as st
import pandas as pd
import altair as alt
import sys
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

st.set_page_config(page_title="Predicciones", page_icon="🔮", layout="wide")
st.title("🔮 Modelado Estocástico y Dinámica del Cambio")

# Carga
df_estaciones = utils.cargar_estaciones()
if df_estaciones.empty: st.stop()

c1, c2 = st.columns(2)
with c1: estado = st.selectbox("Estado:", sorted(df_estaciones['ESTADO'].unique()))
with c2: variable = st.selectbox("Variable:", ["TMAX", "TMIN", "PRECIP"])

with st.spinner("Cargando historia..."):
    df = utils.cargar_lecturas_por_estado(estado)
if df.empty: st.stop()

# Procesamiento
df['FECHA'] = pd.to_datetime(df['FECHA'])
df[variable] = pd.to_numeric(df[variable], errors='coerce')
agg = 'sum' if variable == 'PRECIP' else 'mean'
df_m = df.set_index('FECHA').resample('MS')[variable].agg(agg)
df_m = df_m.reindex(pd.date_range(df_m.index.min(), df_m.index.max(), freq='MS')).interpolate()
df_m = df_m.reset_index()
df_m.columns = ['FECHA', variable]

tab1, tab2 = st.tabs(["📉 Cinemática", "⚙️ Predicción SARIMA"])

with tab1:
    st.subheader("Velocidad de Cambio (Derivada)")
    try:
        decomp = seasonal_decompose(df_m.set_index('FECHA')[variable], period=12)
        trend = decomp.trend
        derivada = np.gradient(trend.dropna()) * 12 # Anualizado
        
        kinematics = pd.DataFrame({
            'Fecha': df_m['FECHA'].iloc[6:-6],
            'Tendencia': trend.dropna().values,
            'Velocidad': derivada
        })
        
        c_trend = alt.Chart(kinematics).mark_line(color='#333').encode(
            x='Fecha:T', y=alt.Y('Tendencia:Q', scale=alt.Scale(zero=False))
        ).properties(title="Tendencia", height=200)
        
        c_vel = alt.Chart(kinematics).mark_bar().encode(
            x='Fecha:T', y='Velocidad:Q',
            color=alt.condition(alt.datum.Velocidad > 0, alt.value("red"), alt.value("blue"))
        ).properties(title="Velocidad (Unidades/Año)", height=200)
        
        st.altair_chart(c_trend & c_vel, theme="streamlit", use_container_width=True)
        
    except: st.warning("Datos insuficientes para cinemática.")

with tab2:
    st.subheader("Predicción (Caja Blanca)")
    if len(df_m) > 36 and st.button("🚀 Calcular"):
        with st.spinner("Optimizando modelo..."):
            try:
                model = pm.auto_arima(df_m[variable], seasonal=True, m=12, suppress_warnings=True)
                pred, conf = model.predict(n_periods=24, return_conf_int=True)
                
                fechas_fut = pd.date_range(df_m['FECHA'].iloc[-1], periods=25, freq='MS')[1:]
                df_fut = pd.DataFrame({'FECHA': fechas_fut, 'PRED': pred.values, 'LOWER': conf[:,0], 'UPPER': conf[:,1]})
                
                # Explicación
                p,d,q = model.order
                P,D,Q,m = model.seasonal_order
                st.info(f"Modelo seleccionado: SARIMA({p},{d},{q})x({P},{D},{Q})12. AIC: {model.aic():.1f}")
                
                # Gráfica
                base = alt.Chart(df_m.tail(96)).mark_line(color='gray').encode(x='FECHA:T', y=alt.Y(variable, scale=alt.Scale(zero=False)))
                line = alt.Chart(df_fut).mark_line(color='red', strokeDash=[5,5]).encode(x='FECHA:T', y='PRED')
                band = alt.Chart(df_fut).mark_area(opacity=0.2, color='red').encode(x='FECHA:T', y='LOWER', y2='UPPER')
                st.altair_chart(base + band + line, theme="streamlit", use_container_width=True)
                
                # Diagnóstico
                res = pd.DataFrame({'Residuos': model.resid()}).reset_index()
                h = alt.Chart(res).mark_bar().encode(x=alt.X('Residuos', bin=True), y='count()').properties(height=200)
                l = alt.Chart(res).mark_line().encode(x='index', y='Residuos').properties(height=200)
                st.altair_chart(h | l, theme="streamlit", use_container_width=True)
                
                # Tabla Segura
                with st.expander("Ver Datos"):
                    df_show = df_fut.copy()
                    df_show['FECHA'] = df_show['FECHA'].dt.strftime('%Y-%m-%d')
                    st.dataframe(df_show.style.format("{:.2f}"), use_container_width=True)
                    
            except Exception as e: st.error(f"Error: {e}")