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
    """Calcula la velocidad de cambio mensual."""
    derivada = serie.diff().fillna(0)
    return derivada

# === MODELO 1: ESTÁNDAR (SARIMA/HW) ===
def generar_modelo_robusto(serie, periodos_deseados=60):
    """
    Modelo Determinista: Busca la mejor línea única.
    """
    historia_disponible = len(serie)
    periodos = min(periodos_deseados, historia_disponible * 2, 60)
    if periodos < 6: periodos = 6
    
    debug_info = {}

    try:
        # Intento 1: SARIMA
        order = (1, 1, 1)
        seasonal = (1, 1, 0, 12)
        
        model = SARIMAX(serie, order=order, seasonal_order=seasonal, 
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False, maxiter=200) 
        forecast = results.get_forecast(steps=periodos)
        
        # Corrección: Renombrar columnas para estandarizar
        conf_df = forecast.conf_int()
        conf_df.columns = ['lower', 'upper'] 
        
        debug_info = {
            "Tipo": "Determinista (Estándar)",
            "Algoritmo": "SARIMA",
            "Orden": str(order),
            "Estacionalidad": str(seasonal),
            "AIC": round(results.aic, 2),
            "Sigma (Error)": round(results.resid.std(), 4)
        }
        
        return forecast.predicted_mean, conf_df, f"SARIMA ({periodos}m)", debug_info
        
    except Exception:
        try:
            # Intento 2: Holt-Winters
            hw_model = ExponentialSmoothing(serie, trend='add', seasonal='add', 
                                            seasonal_periods=min(12, int(len(serie)/2))).fit()
            pred = hw_model.forecast(periodos)
            std = serie.std()
            conf = pd.DataFrame({"lower": pred - 1.96*std, "upper": pred + 1.96*std}, index=pred.index)
            
            debug_info = {
                "Tipo": "Determinista (Estándar)",
                "Algoritmo": "Holt-Winters",
                "AIC": round(hw_model.aic, 2),
                "Nota": "Fallback usado"
            }
            return pred, conf, f"Holt-Winters ({periodos}m)", debug_info
            
        except Exception as e:
            return None, None, str(e), {}

# === MODELO 2: HÍBRIDO (SARIMA + MONTECARLO) ===
def simulacion_montecarlo_hibrido(serie, steps, n_simulaciones=200):
    """
    Modelo Híbrido: Usa SARIMA para la estructura (estaciones) y Montecarlo para el ruido.
    """
    debug_info = {}
    
    # 1. Ajuste Base
    try:
        model = SARIMAX(serie, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), 
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False, maxiter=200)
        forecast_base = model_fit.get_forecast(steps=steps).predicted_mean
        resid = model_fit.resid
        metodo = "SARIMA"
    except:
        try:
            model = ExponentialSmoothing(serie, trend='add', seasonal='add', 
                                         seasonal_periods=min(12, int(len(serie)/2)))
            model_fit = model.fit()
            forecast_base = model_fit.forecast(steps)
            resid = model_fit.resid
            metodo = "HW"
        except Exception as e:
            return None, None, None, f"Error: {str(e)}", {}

    # 2. Montecarlo
    simulaciones = []
    std_resid = resid.std()
    
    debug_info = {
        "Tipo": "Híbrido (Estructura + Azar)",
        "Modelo Base": metodo,
        "Simulaciones": n_simulaciones,
        "Volatilidad Base (Std)": round(std_resid, 4)
    }
    
    for _ in range(n_simulaciones):
        ruido = np.random.normal(0, std_resid, steps)
        escenario = pd.Series(forecast_base + ruido, name=None)
        simulaciones.append(escenario)
    
    df_sims = pd.DataFrame(simulaciones).T 
    df_sims.index = forecast_base.index
    df_sims.columns = [f"sim_{i}" for i in range(df_sims.shape[1])]
    
    proyeccion_mediana = df_sims.median(axis=1)
    limite_inferior = df_sims.quantile(0.05, axis=1) 
    limite_superior = df_sims.quantile(0.95, axis=1)
    conf_int = pd.DataFrame({"lower": limite_inferior, "upper": limite_superior})
    
    return proyeccion_mediana, conf_int, df_sims, f"Híbrido ({metodo}+MC)", debug_info

# === MODELO 3: MONTECARLO PURO (RANDOM WALK) ===
def simulacion_montecarlo_puro(serie, steps, n_simulaciones=200):
    """
    Modelo Ingenuo: Random Walk.
    """
    debug_info = {}
    
    # 1. Calcular estadísticas
    diffs = serie.diff().dropna()
    mu = diffs.mean()
    sigma = diffs.std()
    last_val = serie.iloc[-1]
    
    simulaciones = []
    
    # Generamos fechas futuras
    last_date = serie.index[-1]
    future_dates = pd.date_range(start=last_date, periods=steps+1, freq=serie.index.freq)[1:]
    
    debug_info = {
        "Tipo": "Estocástico Puro (Random Walk)",
        "Supuesto": "Aleatorio basado en volatilidad pasada",
        "Deriva Promedio (Mu)": round(mu, 4),
        "Volatilidad (Sigma)": round(sigma, 4),
        "Simulaciones": n_simulaciones
    }
    
    for _ in range(n_simulaciones):
        random_shocks = np.random.normal(mu, sigma, steps)
        trayectoria = [last_val]
        for shock in random_shocks:
            trayectoria.append(trayectoria[-1] + shock)
        
        sim_series = pd.Series(trayectoria[1:], index=future_dates, name=None)
        simulaciones.append(sim_series)

    df_sims = pd.DataFrame(simulaciones).T
    df_sims.columns = [f"sim_{i}" for i in range(df_sims.shape[1])]
    
    proyeccion_mediana = df_sims.median(axis=1)
    limite_inferior = df_sims.quantile(0.05, axis=1) 
    limite_superior = df_sims.quantile(0.95, axis=1)
    conf_int = pd.DataFrame({"lower": limite_inferior, "upper": limite_superior})
    
    return proyeccion_mediana, conf_int, df_sims, "Montecarlo Puro (RW)", debug_info

# --- VISUALIZACIÓN UI (Helper) ---
def mostrar_caja_blanca(info_dict):
    """Renderiza la tabla de parámetros técnicos."""
    with st.expander("🛠️ Parámetros del Modelo (Caja Blanca)"):
        if info_dict:
            df_params = pd.DataFrame(list(info_dict.items()), columns=["Parámetro", "Valor"])
            # Convertimos a string para evitar errores de PyArrow con tipos mixtos
            df_params["Valor"] = df_params["Valor"].astype(str)
            st.table(df_params)
        else:
            st.write("N/A")

# --- GRÁFICAS ---

def plot_proyeccion_generica(historia, pred_mediana, conf_int, simulaciones_raw, titulo, color_linea, var, mostrar_sims=True):
    """Función genérica para graficar cualquier modelo"""
    fig = go.Figure()
    historia_vis = historia.tail(180) 
    fig.add_trace(go.Scatter(x=historia_vis.index, y=historia_vis.values, mode='lines', name='Histórico', line=dict(color='gray', width=1)))
    
    # Simulaciones (Tenues)
    if mostrar_sims and simulaciones_raw is not None:
        sample_sims = simulaciones_raw.sample(n=min(20, len(simulaciones_raw.columns)), axis=1, random_state=42)
        for col in sample_sims.columns:
            fig.add_trace(go.Scatter(x=sample_sims.index, y=sample_sims[col], mode='lines', 
                                     line=dict(color=color_linea, width=0.5), opacity=0.1, showlegend=False, hoverinfo='skip'))

    # Intervalo (Cotas visibles)
    if conf_int is not None:
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(conf_int.index), pd.Series(conf_int.index[::-1])]),
            y=pd.concat([conf_int["upper"], conf_int["lower"][::-1]]),
            fill='toself', fillcolor=color_linea, opacity=0.25,
            line=dict(width=1, color=color_linea, dash='dot'),
            name='Intervalo Confianza', hoverinfo="skip"
        ))

    # Mediana / Predicción Principal
    fig.add_trace(go.Scatter(x=pred_mediana.index, y=pred_mediana.values, mode='lines', name='Proyección (Mediana)', line=dict(color=color_linea, width=3)))
    
    fig.update_layout(title=titulo, yaxis_title=var, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    return fig

def plot_comparativa_tres_modelos(historia, res_std, res_hib, res_puro, titulo, var):
    """
    Compara: Estándar (Azul), Híbrido (Rojo), Puro (Naranja)
    """
    fig = go.Figure()
    historia_vis = historia.tail(180) 
    fig.add_trace(go.Scatter(x=historia_vis.index, y=historia_vis.values, mode='lines', name='Histórico', line=dict(color='gray', width=1)))

    # 1. ESTÁNDAR (Azul)
    pred_s, conf_s = res_std[0], res_std[1]
    if pred_s is not None:
        col = "#1f77b4" # Azul
        fig.add_trace(go.Scatter(x=pred_s.index, y=pred_s.values, mode='lines', name='Estándar (SARIMA)', line=dict(color=col, width=2, dash='dash')))
        if conf_s is not None:
            fig.add_trace(go.Scatter(x=pd.concat([pd.Series(conf_s.index), pd.Series(conf_s.index[::-1])]),
                                     y=pd.concat([conf_s.iloc[:,1], conf_s.iloc[:,0][::-1]]),
                                     fill='toself', fillcolor=col, opacity=0.1, line=dict(width=0), showlegend=False, hoverinfo='skip'))

    # 2. PURO (Naranja)
    pred_p, conf_p = res_puro[0], res_puro[1]
    if pred_p is not None:
        col = "#ff7f0e" # Naranja
        fig.add_trace(go.Scatter(x=pred_p.index, y=pred_p.values, mode='lines', name='Montecarlo Puro (RW)', line=dict(color=col, width=2)))
        if conf_p is not None:
             fig.add_trace(go.Scatter(x=pd.concat([pd.Series(conf_p.index), pd.Series(conf_p.index[::-1])]),
                                     y=pd.concat([conf_p["upper"], conf_p["lower"][::-1]]),
                                     fill='toself', fillcolor=col, opacity=0.1, line=dict(width=0), showlegend=False, hoverinfo='skip'))

    # 3. HÍBRIDO (Rojo - El mejor)
    pred_h, conf_h = res_hib[0], res_hib[1]
    if pred_h is not None:
        col = "#d62728" # Rojo
        fig.add_trace(go.Scatter(x=pred_h.index, y=pred_h.values, mode='lines', name='Híbrido (SARIMA+MC)', line=dict(color=col, width=3)))
        if conf_h is not None:
             fig.add_trace(go.Scatter(x=pd.concat([pd.Series(conf_h.index), pd.Series(conf_h.index[::-1])]),
                                     y=pd.concat([conf_h["upper"], conf_h["lower"][::-1]]),
                                     fill='toself', fillcolor=col, opacity=0.2, line=dict(width=0), showlegend=False, hoverinfo='skip'))

    fig.update_layout(title=f"Comparativa Total: {titulo}", yaxis_title=var, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    return fig

# --- INTERFAZ ---
st.title("🔮 Laboratorio de Predicciones")

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

# --- CARGA REACTIVA ---
with st.spinner(f"Analizando estaciones de {estado_sel}..."):
    df_raw = utils.cargar_lecturas_por_estado(estado_sel)

if df_raw.empty:
    st.warning("Este estado no tiene registros históricos.")
    st.stop()

# 3. FILTRO CALIDAD
conteo_estaciones = df_raw['ESTACION'].value_counts()
MINIMO_REGISTROS = 1000 
estaciones_validas_ids = conteo_estaciones[conteo_estaciones >= MINIMO_REGISTROS].index.tolist()
df_estaciones_validas = df_estaciones[(df_estaciones['ESTADO'] == estado_sel) & (df_estaciones['ESTACION'].isin(estaciones_validas_ids))]

opciones_estaciones = {}
for _, row in df_estaciones_validas.iterrows():
    cant_datos = conteo_estaciones.get(row['ESTACION'], 0)
    label = f"{row['NOMBRE']} (Regs: {cant_datos})"
    opciones_estaciones[label] = row['ESTACION']

# 4. SELECTORES
col1, col2 = st.columns(2)
with col1:
    if not opciones_estaciones:
        st.warning("⚠️ Sin estaciones con datos suficientes.")
        id_estacion = None
        nombre_sel = "N/A"
    else:
        label_sel = st.selectbox("🏠 2. Estación", list(opciones_estaciones.keys()))
        id_estacion = opciones_estaciones[label_sel]
        nombre_sel = label_sel.split("(")[0] 

with col2:
    var_map = {"Temp. Máxima": "TMAX", "Temp. Mínima": "TMIN", "Lluvia": "PRECIP", "Evaporación": "EVAP"}
    var_txt = st.selectbox("📊 3. Variable", list(var_map.keys()))
    col_db = var_map[var_txt]

# 5. EJECUCIÓN
if st.button("🚀 Ejecutar Experimento de Modelos", type="primary"):
    
    df_raw['FECHA'] = pd.to_datetime(df_raw['FECHA'])
    df_raw[col_db] = pd.to_numeric(df_raw[col_db], errors='coerce')
    PERIODOS_PRED = 60 

    # --- PREPARACIÓN DE DATOS ---
    serie_estatal_diaria = df_raw.groupby("FECHA")[col_db].median().dropna()
    rule = 'MS'
    
    if col_db in ["PRECIP", "EVAP"]:
        serie_estatal = serie_estatal_diaria.resample(rule).sum() 
    else:
        serie_estatal = serie_estatal_diaria.resample(rule).median()
    serie_estatal = serie_estatal.interpolate(method='linear')

    serie_loc = pd.Series(dtype=float)
    if id_estacion:
        df_loc = df_raw[df_raw["ESTACION"] == id_estacion].set_index("FECHA").sort_index()
        serie_loc_diaria = df_loc[col_db].dropna()
        if not serie_loc_diaria.empty:
            if col_db in ["PRECIP", "EVAP"]:
                serie_loc = serie_loc_diaria.resample(rule).sum()
            else:
                serie_loc = serie_loc_diaria.resample(rule).median()
            serie_loc = serie_loc.interpolate(method='linear')

    # --- CÁLCULO DE LOS 3 MODELOS ---
    with st.spinner("Calculando modelos..."):
        
        # A. NIVEL ESTATAL
        res_est_std = generar_modelo_robusto(serie_estatal)
        res_est_hib = simulacion_montecarlo_hibrido(serie_estatal, PERIODOS_PRED)
        res_est_pur = simulacion_montecarlo_puro(serie_estatal, PERIODOS_PRED)
        
        # B. NIVEL LOCAL
        res_loc_std = (None, None, None, {})
        res_loc_hib = (None, None, None, None, {})
        res_loc_pur = (None, None, None, None, {})
        
        if not serie_loc.empty and len(serie_loc) >= 24:
            res_loc_std = generar_modelo_robusto(serie_loc)
            res_loc_hib = simulacion_montecarlo_hibrido(serie_loc, PERIODOS_PRED)
            res_loc_pur = simulacion_montecarlo_puro(serie_loc, PERIODOS_PRED)

    # --- VISUALIZACIÓN ---
    tab1, tab2, tab3, tab4 = st.tabs(["🔵 Estándar (SARIMA)", "🔴 Híbrido (SARIMA+MC)", "🟠 Montecarlo Puro", "🆚 Comparativa Final"])

    # 1. ESTÁNDAR
    with tab1:
        st.markdown("**Modelo Clásico:** Una sola línea de predicción determinista.")
        c1, c2 = st.columns(2)
        with c1:
            if res_est_std[0] is not None:
                st.plotly_chart(plot_proyeccion_generica(serie_estatal, res_est_std[0], res_est_std[1], None, f"Estatal: {res_est_std[2]}", "#1f77b4", var_txt, False), width="stretch")
                mostrar_caja_blanca(res_est_std[3])
        with c2:
            if res_loc_std[0] is not None:
                st.plotly_chart(plot_proyeccion_generica(serie_loc, res_loc_std[0], res_loc_std[1], None, f"Local: {res_loc_std[2]}", "#1f77b4", var_txt, False), width="stretch")
                mostrar_caja_blanca(res_loc_std[3])

    # 2. HÍBRIDO
    with tab2:
        st.markdown("**Modelo Robusto:** Combina la estructura de SARIMA con el riesgo de Montecarlo.")
        c1, c2 = st.columns(2)
        with c1:
            if res_est_hib[0] is not None:
                st.plotly_chart(plot_proyeccion_generica(serie_estatal, res_est_hib[0], res_est_hib[1], res_est_hib[2], f"Estatal: {res_est_hib[3]}", "#d62728", var_txt), width="stretch")
                mostrar_caja_blanca(res_est_hib[4])
        with c2:
            if res_loc_hib[0] is not None:
                st.plotly_chart(plot_proyeccion_generica(serie_loc, res_loc_hib[0], res_loc_hib[1], res_loc_hib[2], f"Local: {res_loc_hib[3]}", "#d62728", var_txt), width="stretch")
                mostrar_caja_blanca(res_loc_hib[4])

    # 3. PURO
    with tab3:
        st.markdown("**Modelo Ingenuo:** Simulación pura basada solo en volatilidad (Random Walk). Observa cómo **pierde las estaciones**.")
        c1, c2 = st.columns(2)
        with c1:
            if res_est_pur[0] is not None:
                st.plotly_chart(plot_proyeccion_generica(serie_estatal, res_est_pur[0], res_est_pur[1], res_est_pur[2], f"Estatal: {res_est_pur[3]}", "#ff7f0e", var_txt), width="stretch")
                mostrar_caja_blanca(res_est_pur[4])
        with c2:
            if res_loc_pur[0] is not None:
                st.plotly_chart(plot_proyeccion_generica(serie_loc, res_loc_pur[0], res_loc_pur[1], res_loc_pur[2], f"Local: {res_loc_pur[3]}", "#ff7f0e", var_txt), width="stretch")
                mostrar_caja_blanca(res_loc_pur[4])

    # 4. COMPARATIVA
    with tab4:
        st.markdown("### 🆚 El Duelo de Modelos")
        
        if res_est_std[0] is not None:
            st.plotly_chart(plot_comparativa_tres_modelos(serie_estatal, res_est_std, res_est_hib, res_est_pur, f"Comparativa Estatal", var_txt), width="stretch")
        
        if res_loc_std[0] is not None:
            st.divider()
            st.plotly_chart(plot_comparativa_tres_modelos(serie_loc, res_loc_std, res_loc_hib, res_loc_pur, f"Comparativa Local", var_txt), width="stretch")

        # --- GLOSARIO CON ANALOGÍAS ---
        st.divider()
        with st.expander("📚 Glosario: Explicación con Analogías (Para no expertos)", expanded=True):
            st.markdown("""
            Para entender cómo funcionan estos modelos matemáticos, imagina que queremos predecir **dónde caerá una flecha**.

            #### 1. 🔵 Modelo Estándar (SARIMA) -> "El Arquero Experto"
            > *Es determinista.*
            El arquero apunta con precisión basándose en su experiencia (el pasado). Sabe que en Diciembre hace frío y en Mayo calor (estaciones).
            * **Lo bueno:** Es muy preciso detectando patrones.
            * **Lo malo:** Asume que el día del disparo no habrá viento inesperado. Si algo raro pasa, falla.

            #### 2. 🟠 Montecarlo Puro (Random Walk) -> "El Caminante Ebrio"
            > *Es estocástico puro (al azar).*
            Imagina una persona que da pasos basándose solo en el paso anterior, pero tambaleándose.
            * **Lo bueno:** Entiende muy bien el caos y el "ruido".
            * **Lo malo:** **No tiene memoria.** Olvida que después del invierno viene la primavera. Por eso la línea naranja suele verse plana o perdida.

            #### 3. 🔴 Modelo Híbrido (SARIMA + MC) -> "El Arquero con Viento" (RECOMENDADO)
            > *Combina lo mejor de los dos mundos.*
            Aquí usamos al **Arquero Experto** para apuntar (captura las estaciones), pero usamos una computadora para simular **200 tipos de viento diferentes** (Montecarlo).
            * El resultado es la línea roja: La trayectoria más probable considerando tanto la habilidad del arquero como el caos del clima.

            #### 4. 🌫️ Intervalo de Confianza (La Sombra) -> "El Haz de Luz"
            Imagina que aluzas un camino oscuro con una linterna. Cerca de ti, el haz de luz es estrecho y ves bien. Mientras más lejos alumbras (más años al futuro), el haz se hace más ancho.
            * La sombra nos dice: *"No estoy seguro del valor exacto, pero hay un 90% de probabilidad de que esté dentro de esta luz"*.
            """)