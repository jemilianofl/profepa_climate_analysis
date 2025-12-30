import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import utils 

# --- CONFIGURACIÓN DE PÁGINA ---
# Esto ayuda a que ocupe todo el ancho
st.set_page_config(page_title="Predicciones Climáticas", layout="wide")

# --- FUNCIONES AUXILIARES (Estas sí deben ir en funciones) ---
def generar_modelo_robusto(serie, periodos=12, variable="TMAX"):
    """
    Intenta generar una predicción usando SARIMA. 
    Si falla por problemas matemáticos (convergencia), usa Holt-Winters.
    """
    # 1. Definir modelo SARIMA (Seasonal ARIMA)
    try:
        model = SARIMAX(serie, 
                        order=(1, 1, 1), 
                        seasonal_order=(1, 1, 0, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        results = model.fit(disp=False, maxiter=500)
        
        forecast = results.get_forecast(steps=periodos)
        prediccion = forecast.predicted_mean
        conf_int = forecast.conf_int()
        conf_int.columns = ["lower", "upper"]
        
        return prediccion, conf_int, "SARIMA (Estadístico Avanzado)"

    except Exception as e:
        # 2. PLAN B: Holt-Winters (Suavizado Exponencial)
        try:
            hw_model = ExponentialSmoothing(serie, 
                                            trend='add', 
                                            seasonal='add', 
                                            seasonal_periods=12).fit()
            prediccion = hw_model.forecast(periodos)
            
            std_dev = serie.std()
            conf_int = pd.DataFrame({
                "lower": prediccion - 1.96 * std_dev,
                "upper": prediccion + 1.96 * std_dev
            }, index=prediccion.index)
            
            return prediccion, conf_int, "Holt-Winters (Suavizado Exponencial)"
            
        except Exception as e2:
            return None, None, f"Error: {str(e2)}"

# --- CÓDIGO PRINCIPAL (Sin indentar, se ejecuta directo) ---
st.title("🔮 Predicciones Climáticas con IA")
st.markdown("""
Este módulo utiliza modelos estadísticos (**SARIMA** y **Holt-Winters**) para proyectar 
el comportamiento futuro del clima basándose en datos históricos.
""")

# --- 1. CARGA DE DATOS ---
with st.spinner("Cargando catálogo de estaciones..."):
    df_estaciones = utils.cargar_estaciones()

if df_estaciones.empty:
    st.error("No hay conexión con la base de datos o la tabla de estaciones está vacía.")
    st.stop() # Detenemos la ejecución si no hay datos

# --- 2. CONTROLES ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    lista_estados = sorted(df_estaciones['ESTADO'].unique())
    estado_sel = st.selectbox("1. Estado", lista_estados)

with col2:
    estaciones_estado = df_estaciones[df_estaciones['ESTADO'] == estado_sel]
    # Crear diccionario nombre -> ID
    mapa_nombres = dict(zip(estaciones_estado['NOMBRE'], estaciones_estado['ESTACION']))
    nombre_sel = st.selectbox("2. Estación", estaciones_estado['NOMBRE'].unique())
    id_estacion = mapa_nombres[nombre_sel]

with col3:
    var_opciones = {
        "Temperatura Máxima (°C)": "TMAX",
        "Temperatura Mínima (°C)": "TMIN",
        "Precipitación (mm)": "PRECIP",
        "Evaporación (mm)": "EVAP"
    }
    variable_sel = st.selectbox("3. Variable", list(var_opciones.keys()))
    col_db = var_opciones[variable_sel]

with col4:
    meses_pred = st.slider("4. Meses a predecir", 6, 24, 12)

# --- 3. PROCESAMIENTO ---
if st.button("🚀 Generar Pronóstico", type="primary"):
    # Carga de datos bajo demanda
    with st.spinner("Consultando base de datos..."):
        df_lecturas = utils.cargar_lecturas_por_estado(estado_sel)
    
    if df_lecturas.empty:
        st.warning("No se encontraron lecturas para este estado.")
        st.stop()

    df_filtrado = df_lecturas[df_lecturas["ESTACION"] == id_estacion].copy()
    
    if df_filtrado.empty:
        st.warning(f"La estación {nombre_sel} no tiene datos históricos registrados.")
        st.stop()

    # Preparación de Series de Tiempo
    df_filtrado['FECHA'] = pd.to_datetime(df_filtrado['FECHA'])
    df_filtrado = df_filtrado.set_index('FECHA').sort_index()
    
    # Conversión a numérico (Corrección del error anterior)
    serie_diaria = pd.to_numeric(df_filtrado[col_db], errors='coerce').dropna()

    if len(serie_diaria) < 365:
        st.error("⚠️ Datos insuficientes: Se necesitan al menos 1 año de registros para predecir.")
        st.stop()

    # --- RESAMPLING ---
    regla_resample = 'MS' # Month Start
    if col_db in ["PRECIP", "EVAP"]:
        serie_mensual = serie_diaria.resample(regla_resample).sum()
    else:
        serie_mensual = serie_diaria.resample(regla_resample).mean()

    # Rellenar huecos
    serie_mensual = serie_mensual.interpolate(method='linear')

    # --- 4. MODELADO ---
    with st.spinner(f"Entrenando modelos para {nombre_sel}..."):
        pred, conf, nombre_modelo = generar_modelo_robusto(serie_mensual, periodos=meses_pred, variable=col_db)

    if pred is None:
        st.error(f"No se pudo generar el modelo. Razón: {nombre_modelo}")
    else:
        st.success(f"✅ Predicción generada exitosamente usando: **{nombre_modelo}**")

        # --- 5. VISUALIZACIÓN ---
        fig = go.Figure()

        # Datos Históricos (Últimos 5 años)
        historia_visible = serie_mensual.tail(60) 
        
        fig.add_trace(go.Scatter(
            x=historia_visible.index, 
            y=historia_visible.values,
            mode='lines',
            name='Histórico (Mensual)',
            line=dict(color='gray', width=1.5)
        ))

        # Predicción
        fig.add_trace(go.Scatter(
            x=pred.index, 
            y=pred.values,
            mode='lines+markers',
            name='Pronóstico',
            line=dict(color='#E63946', width=2.5)
        ))

        # Intervalo de Confianza
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(conf.index), pd.Series(conf.index[::-1])]),
            y=pd.concat([conf['upper'], conf['lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(230, 57, 70, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='Intervalo de Confianza (95%)'
        ))

        fig.update_layout(
            title=f"Proyección de {variable_sel} - {nombre_sel}",
            xaxis_title="Fecha",
            yaxis_title=variable_sel,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1)
        )

        st.plotly_chart(fig, width="stretch")

        # --- 6. MÉTRICAS ---
        try:
            ultimo_valor = float(historia_visible.iloc[-1])
            valor_predicho_final = float(pred.iloc[-1])
            delta = valor_predicho_final - ultimo_valor
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Último Dato Registrado", f"{ultimo_valor:.1f}")
            m2.metric(f"Proyección a {meses_pred} meses", f"{valor_predicho_final:.1f}", f"{delta:.1f}")
            m3.metric("Tendencia Detectada", "📈 Alza" if delta > 0 else "📉 Baja")
        except Exception as e:
            st.warning(f"No se pudieron calcular las métricas finales: {e}")