import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import utils 

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicciones Climáticas", layout="wide")

# --- 1. FUNCIONES MATEMÁTICAS ---

def calcular_derivada(serie):
    """Calcula la velocidad de cambio mensual (Tasa de cambio)."""
    derivada = serie.diff().fillna(0)
    return derivada

def generar_modelo_robusto(serie, periodos_deseados=60):
    historia_disponible = len(serie)
    # Regla: Predecir máximo el doble de la historia disponible, tope 60 meses
    periodos = min(periodos_deseados, historia_disponible * 2, 60)
    if periodos < 6: periodos = 6
    
    try:
        model = SARIMAX(serie, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), 
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False, maxiter=200) 
        forecast = results.get_forecast(steps=periodos)
        return forecast.predicted_mean, forecast.conf_int(), f"SARIMA ({periodos}m)"
    except Exception:
        try:
            hw_model = ExponentialSmoothing(serie, trend='add', seasonal='add', 
                                            seasonal_periods=min(12, int(len(serie)/2))).fit()
            pred = hw_model.forecast(periodos)
            std = serie.std()
            conf = pd.DataFrame({"lower": pred - 1.96*std, "upper": pred + 1.96*std}, index=pred.index)
            return pred, conf, f"Holt-Winters ({periodos}m)"
        except Exception as e:
            return None, None, str(e)

def plot_proyeccion(historia, prediccion, conf_int, titulo, color_linea, var):
    """Gráfica de Línea (Serie de Tiempo)"""
    fig = go.Figure()
    # Mostramos solo los últimos 15 años de historia para que la gráfica sea legible
    historia_vis = historia.tail(180) 
    fig.add_trace(go.Scatter(x=historia_vis.index, y=historia_vis.values, mode='lines', name='Histórico', line=dict(color='gray', width=1)))
    fig.add_trace(go.Scatter(x=prediccion.index, y=prediccion.values, mode='lines', name='Proyección', line=dict(color=color_linea, width=2.5)))
    
    if conf_int is not None:
        fig.add_trace(go.Scatter(x=pd.concat([pd.Series(conf_int.index), pd.Series(conf_int.index[::-1])]),
                                 y=pd.concat([conf_int.iloc[:,1], conf_int.iloc[:,0][::-1]]),
                                 fill='toself', fillcolor=color_linea, opacity=0.15, line=dict(width=0), name='IC 95%', hoverinfo="skip"))
    
    fig.update_layout(title=titulo, yaxis_title=var, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    return fig

def plot_derivada_gradiente(prediccion, titulo):
    """
    Gráfica de Barras con Gradiente Rojo/Azul para la Derivada.
    Rojo = Cambio Positivo (Aumento)
    Azul = Cambio Negativo (Disminución)
    """
    derivada = calcular_derivada(prediccion)
    
    # Lógica de colores condicionales
    colores = []
    for valor in derivada.values:
        if valor >= 0:
            colores.append('#E63946') # Rojo (Aumento)
        else:
            colores.append('#1D3557') # Azul Oscuro (Disminución)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=derivada.index, 
        y=derivada.values, 
        marker_color=colores, # Aplicamos la lista de colores
        name="Tasa de Cambio"
    ))
    
    # Línea cero de referencia
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    
    fig.update_layout(
        title=f"Velocidad de Cambio (Gradiente): {titulo}", 
        yaxis_title="Delta Mensual (Cambio)", 
        template="plotly_white", 
        hovermode="x unified"
    )
    return fig

# --- INTERFAZ DE USUARIO ---
st.title("🔮 Análisis Climático Predictivo")
st.markdown("""
Análisis de tendencias a futuro (hasta 5 años). 
La gráfica de **Velocidad de Cambio** usa colores para indicar si la tendencia es al alza (<span style='color:#E63946'><b>Rojo</b></span>) o a la baja (<span style='color:#1D3557'><b>Azul</b></span>).
""", unsafe_allow_html=True)

# 1. CARGA INICIAL
with st.spinner("Conectando DB..."):
    df_estaciones = utils.cargar_estaciones()

if df_estaciones.empty:
    st.error("❌ Base de datos no disponible.")
    st.stop()

# 2. SELECTOR DE ESTADO
col_state, col_var = st.columns([1, 2])
with col_state:
    lista_estados = sorted(df_estaciones['ESTADO'].unique())
    estado_sel = st.selectbox("📍 1. Selecciona Estado", lista_estados)

# --- CARGA REACTIVA DE DATOS DEL ESTADO ---
with st.spinner(f"Analizando estaciones de {estado_sel}..."):
    df_raw = utils.cargar_lecturas_por_estado(estado_sel)

if df_raw.empty:
    st.warning("Este estado no tiene registros históricos.")
    st.stop()

# 3. FILTRO DE CALIDAD DE ESTACIONES
conteo_estaciones = df_raw['ESTACION'].value_counts()
MINIMO_REGISTROS = 1000 
estaciones_validas_ids = conteo_estaciones[conteo_estaciones >= MINIMO_REGISTROS].index.tolist()

df_estaciones_validas = df_estaciones[
    (df_estaciones['ESTADO'] == estado_sel) & 
    (df_estaciones['ESTACION'].isin(estaciones_validas_ids))
]

opciones_estaciones = {}
for _, row in df_estaciones_validas.iterrows():
    cant_datos = conteo_estaciones.get(row['ESTACION'], 0)
    label = f"{row['NOMBRE']} (Regs: {cant_datos})"
    opciones_estaciones[label] = row['ESTACION']

# 4. RESTO DE SELECTORES
col1, col2 = st.columns(2)
with col1:
    if not opciones_estaciones:
        st.warning("⚠️ Ninguna estación tiene suficientes datos (>3 años) para predicciones individuales.")
        id_estacion = None
        nombre_sel = "N/A"
    else:
        label_sel = st.selectbox("🏠 2. Estación (Filtradas por Calidad)", list(opciones_estaciones.keys()))
        id_estacion = opciones_estaciones[label_sel]
        nombre_sel = label_sel.split("(")[0] 

with col2:
    var_map = {"Temp. Máxima": "TMAX", "Temp. Mínima": "TMIN", "Lluvia": "PRECIP", "Evaporación": "EVAP"}
    var_txt = st.selectbox("📊 3. Variable", list(var_map.keys()))
    col_db = var_map[var_txt]

# 5. BOTÓN DE EJECUCIÓN
if st.button("🚀 Ejecutar Análisis", type="primary"):
    
    # Procesamiento
    df_raw['FECHA'] = pd.to_datetime(df_raw['FECHA'])
    df_raw[col_db] = pd.to_numeric(df_raw[col_db], errors='coerce')

    # --- ANÁLISIS ESTATAL ---
    serie_estatal_diaria = df_raw.groupby("FECHA")[col_db].mean().dropna()
    rule = 'MS'
    if col_db in ["PRECIP", "EVAP"]:
        serie_estatal = serie_estatal_diaria.resample(rule).sum()
    else:
        serie_estatal = serie_estatal_diaria.resample(rule).mean()
    serie_estatal = serie_estatal.interpolate(method='linear')

    # --- ANÁLISIS LOCAL ---
    serie_loc = pd.Series()
    if id_estacion:
        df_loc = df_raw[df_raw["ESTACION"] == id_estacion].set_index("FECHA").sort_index()
        serie_loc_diaria = df_loc[col_db].dropna()
        if not serie_loc_diaria.empty:
            if col_db in ["PRECIP", "EVAP"]:
                serie_loc = serie_loc_diaria.resample(rule).sum()
            else:
                serie_loc = serie_loc_diaria.resample(rule).mean()
            serie_loc = serie_loc.interpolate(method='linear')

    # --- MODELADO ---
    with st.spinner("Calculando proyecciones..."):
        # Estatal
        if len(serie_estatal) >= 12:
            pred_est, conf_est, mod_est = generar_modelo_robusto(serie_estatal)
        else:
            pred_est = None
            st.error(f"Datos estatales insuficientes.")

        # Local
        pred_loc = None
        if not serie_loc.empty and len(serie_loc) >= 12:
            pred_loc, conf_loc, mod_loc = generar_modelo_robusto(serie_loc)

    # --- VISUALIZACIÓN ---
    tab1, tab2 = st.tabs(["🌍 Análisis Estatal", "🏠 Análisis Local"])

    with tab1:
        if pred_est is not None:
            c_a, c_b = st.columns([3, 1])
            with c_a:
                st.success(f"Modelo Estatal: {mod_est}")
                st.plotly_chart(plot_proyeccion(serie_estatal, pred_est, conf_est, f"Tendencia Promedio: {estado_sel}", "#2A9D8F", var_txt), width="stretch")
            with c_b:
                # AQUÍ ESTÁ EL CAMBIO: Usamos plot_derivada_gradiente
                st.plotly_chart(plot_derivada_gradiente(pred_est, "Velocidad Estatal"), width="stretch")
            
            try:
                v_act = float(serie_estatal.iloc[-1])
                v_fut = float(pred_est.iloc[-1])
                st.metric("Promedio Futuro", f"{v_fut:.2f}", f"{v_fut-v_act:.2f}")
            except: pass

    with tab2:
        if pred_loc is not None:
            c_c, c_d = st.columns([3, 1])
            with c_c:
                st.success(f"Modelo Local: {mod_loc}")
                st.plotly_chart(plot_proyeccion(serie_loc, pred_loc, conf_loc, f"Tendencia Local: {nombre_sel}", "#E63946", var_txt), width="stretch")
            with c_d:
                # AQUÍ ESTÁ EL CAMBIO: Usamos plot_derivada_gradiente
                st.plotly_chart(plot_derivada_gradiente(pred_loc, "Velocidad Local"), width="stretch")
            
            try:
                v_act_l = float(serie_loc.iloc[-1])
                v_fut_l = float(pred_loc.iloc[-1])
                st.metric("Valor Futuro", f"{v_fut_l:.2f}", f"{v_fut_l-v_act_l:.2f}")
            except: pass
        else:
            if id_estacion:
                st.warning("La estación seleccionada no tiene suficientes datos continuos.")
            else:
                st.info("No hay estación seleccionada.")