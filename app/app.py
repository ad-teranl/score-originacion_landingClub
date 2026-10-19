"""Tablero del score de originación PD (XGBoost y CatBoost, escenario E3).

Ejecutar desde la carpeta DESPLIEGUE:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from puntuar import (COLUMNAS_CRUDAS, NOMBRES_ES, VARIABLE_DE_WOE, cargar_artefactos, cargar_config,  # noqa: E402
                     contribuciones_cat, contribuciones_xgb, corte_costo, cut_off, puntos_tarjeta, puntuar, validar)
from simular import simular, simular_aleatorio  # noqa: E402

AZUL, ROJO, VERDE, AMBAR, GRIS = "#5c6a97", "#d86e6e", "#4f9d69", "#d9a441", "#8a8f98"
COLOR_CAT = {"Normal": VERDE, "Inusual": AMBAR, "Atípico": ROJO}

DESTINOS = {
    "debt_consolidation": "Consolidación de deudas", "credit_card": "Tarjeta de crédito",
    "home_improvement": "Mejoras del hogar", "major_purchase": "Compra grande", "small_business": "Pequeño negocio",
    "car": "Vehículo", "medical": "Gastos médicos", "moving": "Mudanza", "house": "Vivienda",
    "vacation": "Vacaciones", "wedding": "Boda", "other": "Otro", "otros": "Otros", "educational": "Educación",
    "renewable_energy": "Energía renovable",
}
VIVIENDA = {"RENT": "Arrienda", "MORTGAGE": "Hipotecada", "OWN": "Propia", "otros": "Otra", "OTHER": "Otra"}
ANTIGUEDAD = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years",
              "8 years", "9 years", "10+ years", "NI"]
ANTIGUEDAD_ES = {"< 1 year": "Menos de 1 año", "1 year": "1 año", "10+ years": "10 años o más", "NI": "No informa"}


ESTADOS = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
           "CT": "Connecticut", "DE": "Delaware", "DC": "Distrito de Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawái",
           "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Luisiana",
           "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Míchigan", "MN": "Minnesota", "MS": "Misisipi",
           "MO": "Misuri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "Nuevo Hampshire", "NJ": "Nueva Jersey",
           "NM": "Nuevo México", "NY": "Nueva York", "NC": "Carolina del Norte", "ND": "Dakota del Norte", "OH": "Ohio",
           "OK": "Oklahoma", "OR": "Oregón", "PA": "Pensilvania", "RI": "Rhode Island", "SC": "Carolina del Sur",
           "SD": "Dakota del Sur", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
           "WA": "Washington", "WV": "Virginia Occidental", "WI": "Wisconsin", "WY": "Wyoming", "otros": "Otros estados"}


def estado(x):
    return ESTADOS.get(str(x), str(x))


def antig(x):
    return ANTIGUEDAD_ES.get(x, x.replace(" years", " años"))


st.set_page_config(page_title="Score de originación", page_icon="📊", layout="wide")

# El traductor automático del navegador reescribe el texto y rompe la actualización de la página
# (error "removeChild"). Se declara el idioma y se desactiva la traducción.
import streamlit.components.v1 as components
components.html("""<script>
const d = window.parent.document;
d.documentElement.lang = "es"; d.documentElement.setAttribute("translate", "no");
d.body.setAttribute("translate", "no"); d.body.classList.add("notranslate");
if (!d.querySelector('meta[name="google"]')) { const m = d.createElement("meta"); m.name = "google"; m.content = "notranslate"; d.head.appendChild(m); }
</script>""", height=0)

# ---------------- tema claro / oscuro (oscuro por defecto) ----------------
with st.container(key="selector_tema"):
    _tema = st.radio("Tema", ["☀️", "🌙"], index=1, horizontal=True, label_visibility="collapsed", key="tema")
OSCURO = _tema == "🌙"

CSS = """<style>
.tarjeta{border-left:8px solid var(--c);background:color-mix(in srgb,var(--c) 11%,transparent);padding:14px 18px;border-radius:6px}
.tarjeta .t{font-size:.85rem;opacity:.7}.tarjeta .g{font-size:1.9rem;font-weight:700;color:var(--c)}
.insignia{display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;font-weight:600;font-size:.85rem}

/* más aire arriba: el título sube */
.block-container{padding-top:2.6rem !important}
h1{padding-top:0 !important;margin-top:-.4rem}
[data-testid="stSidebarContent"]{padding-top:0 !important}
[data-testid="stSidebarHeader"]{height:1.6rem !important;min-height:0 !important;padding:0 .5rem !important}

/* menú de pestañas: botones llamativos */
.stTabs [data-baseweb="tab-list"]{gap:.4rem;border-bottom:3px solid #5c6a97;padding-bottom:0}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
.stTabs [data-baseweb="tab"]{height:auto;padding:.65rem .95rem;border-radius:12px 12px 0 0;background:#e6e9f4;border:1px solid #c5cce3;border-bottom:none;
  box-shadow:0 -2px 6px rgba(92,106,151,.12);transition:all .15s}
.stTabs [data-baseweb="tab"] p{font-size:.98rem;font-weight:700;color:#3b4468}
.stTabs [data-baseweb="tab"]:hover{background:#d3d9ee;transform:translateY(-2px)}
.stTabs [aria-selected="true"]{background:#5c6a97 !important;border-color:#5c6a97}
.stTabs [aria-selected="true"] p{color:#fff !important}


/* navegación en forma de botones grandes; conserva la sección al recalcular */
.st-key-navegacion [role="radiogroup"]{gap:.3rem;border-bottom:3px solid #5c6a97;padding-bottom:0;flex-wrap:wrap}
.st-key-navegacion label{margin:0 !important;padding:.6rem .8rem;border-radius:12px 12px 0 0;background:#e6e9f4;border:1px solid #c5cce3;border-bottom:none;
  box-shadow:0 -2px 6px rgba(92,106,151,.12);cursor:pointer;transition:all .15s}
.st-key-navegacion label > div:first-child{display:none}
.st-key-navegacion label > div:last-child{margin-left:0 !important;padding-left:0 !important}
.st-key-navegacion label p{font-size:.93rem;white-space:nowrap;font-weight:700;color:#3b4468}
.st-key-navegacion label:hover{background:#d3d9ee;transform:translateY(-2px)}
.st-key-navegacion label:has(input:checked){background:#5c6a97;border-color:#5c6a97}
.st-key-navegacion label:has(input:checked) p{color:#fff !important}

/* selector de tema arriba a la derecha */
.st-key-selector_tema{position:fixed;top:.55rem;right:7.4rem;z-index:1000002;width:auto !important}
.st-key-selector_tema [role="radiogroup"]{gap:.3rem;background:@FONDO@;padding:.2rem;border-radius:999px;border:1px solid @BORDE@}
.st-key-selector_tema label{margin:0 !important;padding:.1rem .6rem;border-radius:999px;cursor:pointer}
.st-key-selector_tema label > div:first-child{display:none}
.st-key-selector_tema label p{font-size:1.35rem;line-height:1.6}
.st-key-selector_tema label:has(input:checked){background:@SEL@}
.st-key-selector_tema{filter:@REV@}
[data-testid="stSidebar"] [data-testid="stImage"]{display:flex;justify-content:center}
.foto-autor{filter:@REV@;border-radius:50%;display:inline-block}
.bloque-autor{display:flex;align-items:center;justify-content:flex-start;gap:.9rem;margin:.3rem 0 1.6rem .2rem}
.autor{font-family:'Roboto Condensed','Arial Narrow',sans-serif;font-weight:700;font-size:1.4rem;line-height:1.1;letter-spacing:.2px}
.autor .nombre{display:inline-block;background:linear-gradient(to right,@G1@ 0%,#14c650 100%);-webkit-background-clip:text;background-clip:text;
  color:transparent !important;-webkit-text-fill-color:transparent;filter:@REV@}
.autor .grado{font-size:1.05rem;opacity:.75;letter-spacing:1px}
"""
OSC = """
html,body{background:#121212}
.stApp{filter:invert(.92) hue-rotate(180deg)}
"""
_estilo = {"FONDO": "#ffffff" if not OSCURO else "#262b36", "BORDE": "#c5cce3" if not OSCURO else "#4a5164",
           "SEL": "#d3d9ee" if not OSCURO else "#5c6a97", "G1": "#314254" if not OSCURO else "#7aa2cf", "REV": "hue-rotate(180deg) invert(.92)" if OSCURO else "none"}
_css = CSS
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
for _k, _v in _estilo.items():
    _css = _css.replace(f"@{_k}@", _v)
st.markdown(_css + (OSC if OSCURO else "") + "</style>", unsafe_allow_html=True)



# ---------------- recursos (se cargan una sola vez) ----------------
@st.cache_resource(show_spinner="Cargando modelos y la base de clientes…")
def recursos():
    cfg = cargar_config()
    art = cargar_artefactos(cfg)
    import ctypes, gc
    import pyarrow.parquet as pq
    tabla = pq.read_table(RAIZ / cfg["rutas"]["modelos"] / cfg["base"]["archivo"], use_threads=False)
    base = tabla.to_pandas(self_destruct=True, split_blocks=True)
    del tabla
    for c, d in [("id", "int32"), ("revenue", "float32"), ("dti_n", "float32"), ("loan_amnt", "int32"), ("fico_n", "float32")]:
        base[c] = base[c].astype(d)      # las PD se quedan en float64: los decimales importan
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)   # devuelve memoria al sistema (Linux/Render)
    except OSError:
        pass
    pd_ordenado = np.sort(base["pd_xgb"].to_numpy())
    return cfg, art, base, pd_ordenado


@st.cache_data
def referencia_validacion():
    return pd.read_parquet(RAIZ / "models" / "referencia_validacion.parquet")


cfg, art, base, pd_ordenado = recursos()
dec, ref_c, ref_n = cfg["decision"], art["ref_cartera"], art["ref_nosup"]
N = len(base)

# ---------------- barra lateral: cut-off ----------------
with st.sidebar:
    import base64
    _foto = base64.b64encode((RAIZ / "app" / "assets" / "autor.png").read_bytes()).decode()
    st.markdown(f'<div class="bloque-autor"><img class="foto-autor" src="data:image/png;base64,{_foto}" width="84">'
                '<div class="autor"><span class="grado">Mgtr.</span><br><span class="nombre">Adrián<br>Terán</span></div></div>', unsafe_allow_html=True)
    st.markdown("### Punto de corte de aprobación")
    OPC = ["Corte de costo (margen y LGD)", "Umbral F1 (cuaderno 4)", "Manual (deslizador)"]
    modo_txt = st.radio("Criterio", OPC, index=0 if dec["cutoff_defecto"] == "costo" else 1)
    explicacion = st.container()          # el texto se llena más abajo, cuando ya se conoce el corte
    margen = st.slider("Margen financiero (ganancia si el cliente paga)", 0.02, 0.12, float(dec["margen"]), 0.005, format="%.3f",
                       disabled=modo_txt != OPC[0])
    lgd = st.slider("Pérdida si no paga (LGD)", 0.30, 0.90, float(dec["lgd"]), 0.05, format="%.2f",
                    disabled=modo_txt != OPC[0])
    if modo_txt == OPC[2]:
        CUT = st.slider("Cut-off entre 0 y 1 (aprobar si la PD es menor)", 0.0, 1.0, round(corte_costo(margen, lgd), 3), 0.001, format="%.3f")
    elif modo_txt == OPC[1]:
        CUT = cut_off(cfg, "f1")
    else:
        CUT = cut_off(cfg, "costo", margen, lgd)

    # ---- texto que cambia según el criterio elegido ----
    _gan, _per, _pc = margen * 100, lgd * 100, corte_costo(margen, lgd)

    def _fila(p, nombre):
        pagan, nopagan = 100 * (1 - p), 100 * p
        return {"Caso": nombre, "Ganancia": f"{pagan:.1f} × {_gan:.0f} = {pagan * _gan:,.0f}",
                "Pérdida": f"{nopagan:.1f} × {_per:.0f} = {nopagan * _per:,.0f}",
                "Resultado": f"{pagan * _gan - nopagan * _per:+,.0f}"}

    with explicacion:
        st.markdown(f"**Base de todo:** cada crédito es de 100 USD. Si el cliente paga, la entidad gana **{_gan:.0f}**. Si no paga, pierde **{_per:.0f}**.")
        if modo_txt == OPC[0]:
            st.markdown(f"**Corte de costo.** Es la PD donde la ganancia es igual a la pérdida. Por debajo, aprobar deja ganancia; por encima, pérdida.\n\n"
                        f"`p = margen / (margen + LGD) = {margen:.2f} / ({margen:.2f} + {lgd:.2f}) = {_pc:.2%}`")
            st.markdown("**Los deslizadores:** *margen* es la ganancia por cada 100 USD cuando el cliente paga; *LGD* es la parte que se pierde cuando no paga. "
                        "Si el margen sube, el corte sube (se aprueba más). Si la LGD sube, el corte baja (se aprueba menos).")
            st.markdown("Con 100 clientes que tienen la misma PD:")
            st.dataframe(pd.DataFrame([_fila(0.05, "PD 5 %"), _fila(_pc, f"PD {_pc:.2%} (el corte)"), _fila(0.15, "PD 15 %")]),
                         hide_index=True, width="stretch")
        elif modo_txt == OPC[1]:
            _f = _fila(CUT, f"PD {CUT:.1%}")
            st.markdown(f"**Umbral F1.** Es la PD {CUT:.1%} donde el modelo detecta mejor a los malos sin equivocarse demasiado en los buenos (F1 máximo en validación, cuaderno 4). "
                        "No mira el dinero, por eso los deslizadores de margen y LGD quedan apagados.")
            st.markdown(f"Con 100 clientes en el borde de este corte (PD {CUT:.1%}), usando margen {margen:.0%} y LGD {lgd:.0%}:")
            st.dataframe(pd.DataFrame([_f]), hide_index=True, width="stretch")
            st.markdown(f"Este corte está {'por encima' if CUT > _pc else 'por debajo'} del punto de equilibrio ({_pc:.2%}): "
                        f"{'aprueba clientes que en promedio hacen perder dinero.' if CUT > _pc else 'es más estricto que el equilibrio.'}")
        else:
            _f = _fila(CUT, f"PD {CUT:.1%}")
            st.markdown("**Manual.** El corte se elige entre 0 y 1 (0.10 = 10 %) y todo se recalcula: la decisión, los gráficos, la lista de los 30 y el cuadro de ganancia.")
            st.markdown(f"Con 100 clientes en el borde del corte elegido (PD {CUT:.1%}), usando margen {margen:.0%} y LGD {lgd:.0%}:")
            st.dataframe(pd.DataFrame([_f]), hide_index=True, width="stretch")
            st.markdown(f"Punto de equilibrio de referencia: {_pc:.2%}. "
                        f"{'El corte elegido está por encima: aprobar en el borde deja pérdida.' if CUT > _pc else 'El corte elegido está en el equilibrio o por debajo: aprobar en el borde no pierde.'}")
        st.caption("Se usa el mismo corte para los dos modelos.")
    st.markdown(f"**Se aprueba si la PD es menor que {CUT:.2%}**")
    st.divider()
    st.markdown("**¿Cuánto se gana o se pierde con este cut-off?**")
    _r = referencia_validacion()
    _ap = _r["pd_calibrada"] < CUT
    _y, _m = _r["default"], _r["loan_amnt"]
    _g = (((1 - _y) * margen - _y * lgd) * _m)[_ap].sum()
    st.metric("Resultado esperado (USD)", f"{_g / 1e6:+,.1f} M", help="Sobre las 123,633 solicitudes de validación, con el margen y la LGD de arriba: margen sobre los que pagan, menos LGD sobre los que no.")
    st.caption(f"Se aprueba el {_ap.mean():.1%}. De los aprobados, no paga el {_y[_ap].mean() if _ap.any() else 0:.1%}. "
               f"Ganancia si pagan: {((1 - _y) * margen * _m)[_ap].sum() / 1e6:,.1f} M. Pérdida si no pagan: {(_y * lgd * _m)[_ap].sum() / 1e6:,.1f} M.")
    st.divider()
    st.caption(f"Modelos: XGBoost (campeón) y CatBoost (retador), escenario {cfg['proyecto']['escenario']} · versión {cfg['proyecto']['version_modelo']}")

st.title("Score de originación PD")
st.caption("Probabilidad de que un solicitante no pague su crédito, con la decisión sugerida, su explicación y un contraste no supervisado.")


# ---------------- piezas de gráficos ----------------
def valor_legible(tratado, col):
    v = tratado[col]
    if col == "revenue_w":
        return f"{v:,.0f}"
    if col == "dti_n_imp":
        return f"{v:.1f}"
    if col == "loan_amnt":
        return f"{v:,.0f}"
    if col == "fico_n":
        return f"{v:.0f}"
    if col == "loan_to_income":
        return f"{v:.2f}"
    if col == "purpose_g":
        return DESTINOS.get(v, v)
    if col == "home_ownership_n_g":
        return VIVIENDA.get(v, v)
    if col == "emp_length":
        return antig(v)
    if col == "addr_state_g":
        return estado(v)
    return str(v)


def g_termometro(pdx, pdc, cut):
    tope = max(0.3, 1.2 * max(pdx, pdc, cut))
    fig = go.Figure(go.Bar(x=[pdx, pdc], y=["XGBoost", "CatBoost"], orientation="h",
                           marker_color=[ROJO if p >= cut else VERDE for p in (pdx, pdc)],
                           text=[f"{p:.4%}" for p in (pdx, pdc)], textposition="outside"))
    fig.add_vline(x=cut, line_dash="dash", line_color=GRIS, annotation_text=f"cut-off {cut:.2%}", annotation_position="top")
    fig.update_layout(height=230, margin=dict(l=10, r=40, t=30, b=30), xaxis=dict(range=[0, tope], tickformat=".0%"),
                      xaxis_title="Probabilidad de no pagar (PD)", yaxis_title=None, bargap=0.45)
    return fig


def g_cartera(pdx, cut):
    h = ref_c["hist_xgb"]
    b = np.array(h["bordes"])
    centros = (b[:-1] + b[1:]) / 2
    fig = go.Figure(go.Bar(x=centros, y=h["cuentas"], marker_color=AZUL, opacity=0.55, width=np.diff(b)))
    fig.add_vline(x=pdx, line_color=ROJO, line_width=3, annotation_text="este cliente", annotation_position="top")
    fig.add_vline(x=cut, line_dash="dash", line_color=GRIS, annotation_text="cut-off", annotation_position="top left")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=30), xaxis=dict(tickformat=".0%"),
                      xaxis_title="PD (XGBoost) de toda la cartera", yaxis_title="Solicitudes", showlegend=False)
    return fig


def g_cascada(aportes, base_v, pd_final_raw, titulo, tratado, key):
    orden = aportes.abs().sort_values(ascending=False).index
    etiquetas = [f"{NOMBRES_ES[c]} = {valor_legible(tratado, VARIABLE_DE_WOE[c])}" for c in orden]
    medidas = ["absolute"] + ["relative"] * len(orden) + ["total"]
    xs = [base_v] + [float(aportes[c]) for c in orden] + [0]
    textos = [f"{base_v:+.2f}"] + [f"{aportes[c]:+.3f}" for c in orden] + [f"{base_v + aportes.sum():+.2f}"]
    fig = go.Figure(go.Waterfall(orientation="h", measure=medidas, y=["Cliente promedio"] + etiquetas + ["Este cliente"],
                                 x=xs, text=textos, textposition="outside",
                                 increasing=dict(marker=dict(color=ROJO)), decreasing=dict(marker=dict(color=AZUL)),
                                 totals=dict(marker=dict(color=GRIS)), connector=dict(line=dict(color="#bbb"))))
    fig.update_layout(title=dict(text=titulo, font=dict(size=14)), height=420, margin=dict(l=10, r=50, t=40, b=30),
                      xaxis_title="Riesgo en log-odds (rojo sube, azul baja)", yaxis=dict(autorange="reversed"), showlegend=False)
    return fig


def g_perfil(tratado):
    cuant = ref_c["cuantiles"]
    vals = {"Ingreso anual": ("revenue", tratado["revenue_w"]), "Deuda / ingreso (DTI)": ("dti_n", tratado["dti_n_imp"]),
            "Monto solicitado": ("loan_amnt", tratado["loan_amnt"]), "Puntaje de buró": ("fico_n", tratado["fico_n"]),
            "Préstamo / ingreso": ("loan_to_income", tratado["loan_to_income"])}
    nombres, pcts, textos = [], [], []
    for nombre, (clave, v) in vals.items():
        p = float(np.interp(v, cuant[clave], np.arange(101)))
        nombres.append(nombre); pcts.append(p)
        textos.append(f"{v:,.2f}".rstrip("0").rstrip(".") + f"  ·  percentil {p:.0f}")
    fig = go.Figure(go.Bar(x=pcts, y=nombres, orientation="h", marker_color=AZUL, opacity=0.8, text=textos,
                           textposition="inside", insidetextanchor="start", textfont=dict(color="white")))
    fig.add_vline(x=50, line_dash="dot", line_color=GRIS, annotation_text="mediana", annotation_position="top")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=30), xaxis=dict(range=[0, 100]),
                      xaxis_title="Posición frente a toda la cartera (0 = el más bajo, 100 = el más alto)",
                      yaxis=dict(autorange="reversed"))
    return fig


def tabla_tramos(tratado, woe_fila):
    tw = art["tabla_woe"]
    filas = []
    for c in NOMBRES_ES:
        var = VARIABLE_DE_WOE[c]
        sub = tw[tw["variable"] == var]
        if var in ("purpose_g", "home_ownership_n_g", "addr_state_g", "emp_length"):
            r = sub[sub["intervalo"] == str(tratado[var])]
            if r.empty:
                r = sub.iloc[(sub["woe"] - woe_fila[c]).abs().argsort()[:1]]
        else:
            r = sub.iloc[(sub["woe"] - woe_fila[c]).abs().argsort()[:1]]
        r = r.iloc[0]
        filas.append({"Variable": NOMBRES_ES[c], "Valor": valor_legible(tratado, var), "Tramo": estado(r["intervalo"]) if var == "addr_state_g" else r["intervalo"],
                      "Default histórico del tramo": float(r["tasa_default"]), "Clientes en el tramo": int(r["n"]),
                      "WoE": float(r["woe"])})
    return pd.DataFrame(filas)


def g_historia(df):
    prom = ref_c["tasa_default_entrenamiento"]
    fig = go.Figure(go.Bar(x=df["Default histórico del tramo"], y=[f"{a} = {b}" for a, b in zip(df["Variable"], df["Valor"])],
                           orientation="h", marker_color=[ROJO if t > prom else AZUL for t in df["Default histórico del tramo"]],
                           text=[f"{t:.1%}" for t in df["Default histórico del tramo"]], textposition="outside"))
    fig.add_vline(x=prom, line_dash="dash", line_color=GRIS, annotation_text=f"promedio {prom:.1%}", annotation_position="top")
    fig.update_layout(height=340, margin=dict(l=10, r=50, t=30, b=30), xaxis=dict(tickformat=".0%"),
                      xaxis_title="Default real de los clientes de entrenamiento que estaban en el mismo tramo",
                      yaxis=dict(autorange="reversed"))
    return fig


def g_puntos(det):
    s = det.iloc[0].sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=[NOMBRES_ES[c] for c in s.index], orientation="h", marker_color=AZUL,
                           text=[f"{v:.1f}" for v in s.values], textposition="outside"))
    fig.update_layout(height=340, margin=dict(l=10, r=40, t=10, b=30), xaxis_title="Puntos que aporta cada variable (más puntos = menos riesgo)")
    return fig


def g_nosup(clave, score, titulo):
    r = ref_n[clave]
    b = np.array(r["bordes"])
    centros = (b[:-1] + b[1:]) / 2
    fig = go.Figure(go.Bar(x=centros, y=r["cuentas"], marker_color=AZUL, opacity=0.45, width=np.diff(b)))
    x = float(np.clip(score, b[0], b[-1]))
    fig.add_vline(x=r["p90"], line_dash="dot", line_color=AMBAR, annotation_text="p90", annotation_position="top left")
    fig.add_vline(x=r["p99"], line_dash="dash", line_color=ROJO, annotation_text="p99", annotation_position="top right")
    fig.add_vline(x=x, line_color="black", line_width=3)
    fig.update_layout(height=200, margin=dict(l=5, r=5, t=25, b=25), showlegend=False, xaxis_title=titulo, yaxis_visible=False)
    return fig


def insignia(cat):
    return f'<span class="insignia" style="background:{COLOR_CAT[cat]}">{cat}</span>'


def ficha_cliente(crudo_fila, t, n_cliente=None, id_original=None):
    """Descripción del solicitante: datos originales en español y variables construidas a partir de ellos."""
    r = crudo_fila.iloc[0]
    tr = cfg["tratamiento"]
    rev, dti_raw = pd.to_numeric(r["revenue"]), pd.to_numeric(r["dti_n"], errors="coerce")
    monto, fico = pd.to_numeric(r["loan_amnt"]), pd.to_numeric(r["fico_n"])
    destino, viv = DESTINOS.get(str(r["purpose"]), str(r["purpose"])), VIVIENDA.get(str(r["home_ownership_n"]), str(r["home_ownership_n"]))
    texto = (f"Solicitante con ingreso anual de **{rev:,.0f} USD** que pide **{monto:,.0f} USD** para **{destino.lower()}**. "
             f"Puntaje de buró **{fico:.0f}**, deuda sobre ingreso (DTI) {('de **' + format(dti_raw, '.1f') + ' %**') if pd.notna(dti_raw) else '**sin dato**'}, vivienda **{viv.lower()}**, "
             f"antigüedad laboral **{antig(str(r['emp_length'])).lower()}** y domicilio en **{estado(r['addr_state'])}**. "
             f"El préstamo equivale a **{t['loan_to_income']:.0%}** de su ingreso anual.")
    st.markdown("##### Quién es el solicitante")
    st.markdown(texto)
    if n_cliente is not None:
        st.caption(f"N.º de cliente {int(n_cliente):,} · ID original {id_original}. En una entidad real, aquí aparecerían también el nombre y la cédula.")
    orig = pd.DataFrame([
        ("Ingreso anual (revenue)", f"{rev:,.0f} USD"), ("Deuda sobre ingreso, DTI (dti_n)", f"{dti_raw:.2f} %" if pd.notna(dti_raw) else "Sin dato"),
        ("Monto solicitado (loan_amnt)", f"{monto:,.0f} USD"), ("Puntaje de buró (fico_n)", f"{fico:.0f}"),
        ("Antigüedad laboral (emp_length)", antig(str(r["emp_length"]))), ("Destino del crédito (purpose)", destino),
        ("Situación de vivienda (home_ownership_n)", viv), ("Estado de residencia (addr_state)", estado(r["addr_state"])),
    ], columns=["Variable original", "Valor declarado"])
    obs_rev = ("Recortado al rango del entrenamiento" if t["revenue_w"] != rev else "Sin cambios")
    obs_dti = ("Sin dato o inválido: se usó la mediana" if (pd.isna(dti_raw) or dti_raw > tr["dti_centinela"]) else "Sin cambios")
    obs_cat = lambda v, orig_v: "Categoría poco frecuente o nueva: agrupada en «otros»" if (str(v) == "otros" and str(orig_v) != "otros") else "Sin cambios"
    cons = pd.DataFrame([
        ("Ingreso ajustado (revenue_w)", f"{t['revenue_w']:,.0f} USD", obs_rev),
        ("DTI imputado (dti_n_imp)", f"{t['dti_n_imp']:.2f} %", obs_dti),
        ("Monto solicitado (loan_amnt)", f"{t['loan_amnt']:,.0f} USD", "Sin cambios"),
        ("Puntaje de buró (fico_n)", f"{t['fico_n']:.0f}", "Sin cambios"),
        ("Préstamo sobre ingreso (loan_to_income)", f"{t['loan_to_income']:.2f}", "Construida: monto solicitado ÷ ingreso ajustado"),
        ("Destino agrupado (purpose_g)", DESTINOS.get(str(t["purpose_g"]), str(t["purpose_g"])), obs_cat(t["purpose_g"], r["purpose"])),
        ("Vivienda agrupada (home_ownership_n_g)", VIVIENDA.get(str(t["home_ownership_n_g"]), str(t["home_ownership_n_g"])), obs_cat(t["home_ownership_n_g"], r["home_ownership_n"])),
        ("Estado agrupado (addr_state_g)", estado(t["addr_state_g"]), obs_cat(t["addr_state_g"], r["addr_state"])),
        ("Antigüedad laboral (emp_length)", antig(str(t["emp_length"])), "Sin cambios"),
    ], columns=["Variable que usa el modelo", "Valor", "Qué pasó con ella"])
    c1, c2 = st.columns([1, 1.6])
    c1.markdown("**Datos originales (8)**")
    c1.dataframe(orig, hide_index=True, width="stretch")
    c2.markdown("**Variables construidas para el modelo (9)**")
    c2.dataframe(cons, hide_index=True, width="stretch")


# ---------------- panel completo de un cliente ----------------
def panel_cliente(crudo_fila, etiqueta, clave, n_cliente=None, id_original=None):
    """crudo_fila: DataFrame de una fila con las 8 columnas crudas. Muestra veredicto, gráficos y contraste no supervisado."""
    salida, tratado, woe = puntuar(crudo_fila, cfg, art, CUT)
    f, t = salida.iloc[0], tratado.iloc[0]
    pdx, pdc = float(f["pd_xgb"]), float(f["pd_cat"])
    aprobado = bool(f["aprobado"])
    color = VERDE if aprobado else ROJO

    st.markdown(f"#### {etiqueta}")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        st.markdown(f"""<div class="tarjeta" style="--c:{color}"><div class="t">Decisión sugerida (XGBoost frente al cut-off de {CUT:.2%})</div>
            <div class="g">{f['decision']}</div><div>{f['puntos_tarjeta']:.0f} puntos en la tarjeta</div></div>""", unsafe_allow_html=True)
    for col, nombre, p in [(c2, "XGBoost", pdx), (c3, "CatBoost", pdc)]:
        with col:
            st.metric(f"PD {nombre}", f"{p:.4%}", f"{'por encima' if p >= CUT else 'por debajo'} del cut-off", delta_color="inverse" if p >= CUT else "normal")
            st.caption(f"Probabilidad de pagar: {1 - p:.4%}")
    if not f["modelos_coinciden"]:
        st.warning("Los dos modelos no coinciden en la decisión: conviene revisar este caso.")

    ficha_cliente(crudo_fila, t, n_cliente, id_original)

    pct_menor = np.searchsorted(pd_ordenado, pdx) / N
    st.markdown("##### Dónde está frente al cut-off y frente a la cartera")
    g1, g2 = st.columns(2)
    g1.plotly_chart(g_termometro(pdx, pdc, CUT), key=f"term_{clave}")
    g2.plotly_chart(g_cartera(pdx, CUT), key=f"cart_{clave}")
    st.caption(f"El {pct_menor:.1%} de las {N:,} solicitudes de la base tiene una PD menor que la de este cliente.")

    st.markdown("##### Por qué cada modelo da esa PD")
    ax, bx = contribuciones_xgb(woe, art)
    ac, bc = contribuciones_cat(woe, art)
    s1, s2 = st.columns(2)
    s1.plotly_chart(g_cascada(ax.iloc[0], bx, pdx, "XGBoost (SHAP exacto)", t, clave), key=f"sx_{clave}")
    s2.plotly_chart(g_cascada(ac.iloc[0], bc, pdc, "CatBoost (SHAP nativo)", t, clave), key=f"sc_{clave}")
    st.caption("Parte del riesgo de un cliente promedio y suma o resta variable por variable. Está en log-odds, antes de la calibración; "
               "si los dos modelos empujan igual, la decisión es robusta.")

    st.markdown("##### Cómo es este cliente frente a la cartera y a la historia")
    p1, p2 = st.columns(2)
    p1.markdown("**Su posición en cada dato numérico**")
    p1.plotly_chart(g_perfil(t), key=f"perf_{clave}")
    tr = tabla_tramos(t, woe.iloc[0])
    p2.markdown("**Qué pasó con clientes parecidos (mismo tramo)**")
    p2.plotly_chart(g_historia(tr), key=f"hist_{clave}")
    q1, q2 = st.columns(2)
    q1.markdown("**Tarjeta de puntos**")
    _, det = puntos_tarjeta(woe, art, detalle=True)
    q1.plotly_chart(g_puntos(det), key=f"pts_{clave}")
    q2.markdown("**Tramos que usó el modelo**")
    q2.dataframe(tr.assign(**{"Default histórico del tramo": (tr["Default histórico del tramo"] * 100).round(1)}).rename(
        columns={"Default histórico del tramo": "Default del tramo (%)"})[["Variable", "Valor", "Tramo", "Default del tramo (%)", "WoE"]],
        hide_index=True, width="stretch")

    st.markdown("##### Contraste no supervisado (sugerencia, no cambia la decisión)")
    st.caption("Los tres detectores no predicen la PD: dicen si el perfil es común o raro frente a los clientes de entrenamiento. "
               "Normal: hasta el percentil 90. Inusual: entre el 90 y el 99. Atípico: por encima del 99.")
    u1, u2, u3 = st.columns(3)
    nombres = {"if": "Isolation Forest", "gmm": "Mezcla gaussiana (GMM)", "ecod": "ECOD"}
    for col, clave_n in zip((u1, u2, u3), ("if", "gmm", "ecod")):
        with col:
            cat = f[f"cat_{clave_n}"]
            st.markdown(f"**{nombres[clave_n]}**  {insignia(cat)}", unsafe_allow_html=True)
            st.plotly_chart(g_nosup(clave_n, f[f"score_{clave_n}"], "puntaje (más alto = más raro)"), key=f"ns_{clave_n}_{clave}")
            if clave_n == "gmm":
                sg = ref_n["segmentos_gmm"][str(int(f["segmento_gmm"]))]
                st.caption(f"Se parece al segmento {int(f['segmento_gmm']) + 1} de {ref_n['componentes']}, donde el default histórico fue {sg['tasa_default']:.1%}.")
    n_at = int(sum(f[f"cat_{c}"] == "Atípico" for c in ("if", "gmm", "ecod")))
    n_in = int(sum(f[f"cat_{c}"] == "Inusual" for c in ("if", "gmm", "ecod")))
    if aprobado and n_at == 0 and n_in == 0:
        msg, nivel = "Los modelos aprueban y ningún detector ve algo raro: la decisión es firme.", st.success
    elif aprobado and n_at == 0:
        msg, nivel = f"Los modelos aprueban. {n_in} de 3 detectores ven el perfil algo inusual: señal leve, conviene mirarlo.", st.info
    elif aprobado:
        msg, nivel = f"Los modelos aprueban, pero {n_at} de 3 detectores ven el perfil atípico: se sugiere revisión manual antes de desembolsar.", st.warning
    elif n_at == 0:
        msg, nivel = "Los modelos no aprueban y el perfil es común: el riesgo viene de los datos del cliente, no de un perfil raro.", st.info
    else:
        msg, nivel = f"Los modelos no aprueban y además {n_at} de 3 detectores ven el perfil atípico: conviene verificar que los datos declarados sean correctos.", st.warning
    nivel(msg)
    if f["fuera_soporte_dti"]:
        st.warning(f"El DTI ({crudo_fila['dti_n'].iloc[0]}) supera el máximo visto en el entrenamiento ({cfg['gobernanza']['dti_max_entrenamiento']}): la PD es menos confiable.")
    if t["revenue_w"] != pd.to_numeric(crudo_fila["revenue"]).iloc[0]:
        st.caption(f"El ingreso se recortó al rango del entrenamiento ({cfg['tratamiento']['revenue_tope_inferior']:,.0f} a {cfg['tratamiento']['revenue_tope_superior']:,.0f} USD).")


def fila_cruda(n):
    r = base.iloc[n - 1]
    d = {c: r[c] for c in COLUMNAS_CRUDAS}
    for c in ["emp_length", "purpose", "home_ownership_n", "addr_state"]:
        d[c] = str(d[c])
    return pd.DataFrame([d], index=[int(n)])


# ---------------- tabla de lote (misma para pestañas 2 y 3) ----------------
def tabla_lote(df, col_id, cut, piso=0.0):
    """df con pd_xgb, pd_cat, cat_if, cat_gmm, cat_ecod, segmento_gmm. Devuelve el DataFrame listo para mostrar."""
    v = pd.DataFrame({
        col_id[0]: col_id[1],
        "PD XGBoost (%)": df["pd_xgb"].to_numpy() * 100,
        "Cercanía al cut-off": np.clip((df["pd_xgb"].to_numpy() - piso) / (cut - piso), 0, 1.0) * 100,
        "PD CatBoost (%)": df["pd_cat"].to_numpy() * 100,
        "Isolation Forest": df["cat_if"].to_numpy(),
        "GMM": [f"{c} · seg. {int(s) + 1}" for c, s in zip(df["cat_gmm"], df["segmento_gmm"])],
        "ECOD": df["cat_ecod"].to_numpy(),
    })
    return v


CONFIG_COLS = {
    "PD XGBoost (%)": st.column_config.NumberColumn("PD XGBoost (%)", format="%.6f"),
    "PD CatBoost (%)": st.column_config.NumberColumn("PD CatBoost (%)", format="%.6f"),
    "Cercanía al cut-off": st.column_config.ProgressColumn("Cercanía al cut-off", min_value=0, max_value=100, format="%.0f%%", help="100 % = justo en el cut-off"),
}


@st.cache_data(show_spinner=False)
def criticos(cut, k):
    pdx = base["pd_xgb"].to_numpy()
    arr = np.where(pdx < cut, pdx, -1.0)
    cand = np.argpartition(-arr, min(k * 10, len(arr) - 1))[: k * 10]
    cand = cand[arr[cand] >= 0]
    sif = base["score_if"].to_numpy()[cand]
    orden = np.lexsort((base["n_cliente"].to_numpy()[cand], -sif, -base["pd_cat"].to_numpy()[cand], -pdx[cand]))
    idx = cand[orden][:k]
    sel = base.iloc[idx].copy()
    sel["empata"] = [int((pdx == v).sum()) for v in sel["pd_xgb"]]
    return sel


PESTANAS = ["Consultar un cliente", "Vigilancia: los 30 más críticos", "Ingesta de nuevos clientes", "Política y sensibilidad", "Sobre el modelo"]
with st.container(key="navegacion"):
    pest = st.radio("Sección", PESTANAS, horizontal=True, label_visibility="collapsed", key="pestana")

# ---------------- 1. consulta por número de cliente ----------------
if pest == PESTANAS[0]:
    if "n_consulta" not in st.session_state:
        st.session_state["n_consulta"] = 1
    a, b, c = st.columns([1.2, 1, 2])
    n = a.number_input(f"N.º de cliente (de 1 a {N:,})", min_value=1, max_value=N, step=1, key="n_consulta")
    b.write("")
    b.button("Cliente al azar", on_click=lambda: st.session_state.update(n_consulta=int(np.random.default_rng().integers(1, N + 1))))
    c.markdown(f"**{cfg['base']['columna_id'].upper()} original:** {int(base.iloc[int(n) - 1]['id'])}")
    c.caption("En una entidad real este campo sería la cédula. La base completa tiene las "
              f"{N:,} solicitudes del archivo original, sin distinguir entrenamiento, validación o fuera de tiempo.")
    panel_cliente(fila_cruda(int(n)), f"Cliente N.º {int(n):,}", f"c{int(n)}", int(n), int(base.iloc[int(n) - 1]["id"]))

    with st.expander("Simular un cliente a mano"):
        with st.form("solicitud"):
            f1, f2, f3, f4 = st.columns(4)
            revenue = f1.number_input("Ingreso anual (USD)", 1000, 2_000_000, 60000, 1000)
            loan_amnt = f2.number_input("Monto solicitado (USD)", 500, 40000, 12000, 500)
            fico = f3.number_input("Puntaje de buró", 600, 850, 700, 1)
            dti = f4.number_input("Deuda / ingreso, DTI (%)", 0.0, 100.0, 18.0, 0.5)
            f5, f6, f7, f8 = st.columns(4)
            purpose = f5.selectbox("Destino", list(DESTINOS)[:12], format_func=DESTINOS.get)
            vivienda = f6.selectbox("Vivienda", ["RENT", "MORTGAGE", "OWN"], format_func=VIVIENDA.get)
            emp = f7.selectbox("Antigüedad laboral", ANTIGUEDAD, index=10, format_func=antig)
            estados = cfg["tratamiento"]["niveles_validos"]["addr_state"]
            estado_sel = f8.selectbox("Estado", estados, index=estados.index("CA"), format_func=estado)
            enviado = st.form_submit_button("Calcular", type="primary")
        if enviado:
            manual = pd.DataFrame([{"revenue": revenue, "dti_n": dti, "loan_amnt": loan_amnt, "fico_n": fico, "emp_length": emp,
                                    "purpose": purpose, "home_ownership_n": vivienda, "addr_state": estado_sel}], index=["manual"])
            panel_cliente(manual, "Cliente simulado a mano", "manual")

# ---------------- 2. vigilancia ----------------
if pest == PESTANAS[1]:
    K = int(dec["top_criticos"])
    pdx_all = base["pd_xgb"].to_numpy()
    n_apr = int((pdx_all < CUT).sum())
    st.markdown(f"De las **{N:,}** solicitudes de la base, **{n_apr:,}** quedan por debajo del cut-off de {CUT:.2%} (se aprobarían) y "
                f"**{N - n_apr:,}** lo superan. Aquí están las **{K} aprobadas que más se acercan al umbral**: las que hay que mirar con cuidado.")
    top = criticos(CUT, K)
    s_top, _, _ = puntuar(pd.DataFrame({c: [str(x) if c in ("emp_length", "purpose", "home_ownership_n", "addr_state") else x for x in top[c]]
                                         for c in COLUMNAS_CRUDAS}, index=top["n_cliente"].to_numpy()), cfg, art, CUT)
    s_top["n_cliente"] = top["n_cliente"].to_numpy()
    s_top["pd_xgb"] = top["pd_xgb"].to_numpy()      # PD exacta ya calculada en la base (todos los decimales)
    s_top["pd_cat"] = top["pd_cat"].to_numpy()
    m1, m2, m3 = st.columns(3)
    m1.metric("PD más baja de la lista", f"{top['pd_xgb'].min():.4%}")
    m2.metric("Distancia del primero al cut-off", f"{(CUT - top['pd_xgb'].iloc[0]) * 100:.6f} pp")
    m3.metric("Clientes con la PD del último puesto", f"{int(top['empata'].iloc[-1]):,}")
    st.warning(f"Solicitudes con alto riesgo: aún se aprobarían, pero están a menos de {(CUT - top['pd_xgb'].iloc[-1]) * 100:.6f} puntos porcentuales del cut-off. "
               "Un pequeño cambio en sus datos las pasa a no aprobadas.")
    t2 = tabla_lote(s_top, ("N.º cliente", top["n_cliente"].to_numpy()), CUT, piso=float(top["pd_xgb"].min()) - (CUT - float(top["pd_xgb"].min())))
    t2.insert(1, "ID original", top["id"].to_numpy())
    t2.insert(0, "Puesto", np.arange(1, len(t2) + 1))
    t2["Empata con"] = top["empata"].to_numpy()
    t2["Ingreso"] = top["revenue"].to_numpy(); t2["DTI"] = top["dti_n"].to_numpy()
    t2["Puntaje de buró"] = top["fico_n"].to_numpy(); t2["Monto"] = top["loan_amnt"].to_numpy()
    st.dataframe(t2, hide_index=True, width="stretch", column_config=CONFIG_COLS, height=min(900, 38 * (len(t2) + 1)))
    st.caption("Orden: PD de XGBoost con todos sus decimales, de mayor a menor. Si dos clientes tienen exactamente la misma PD (mismo perfil en las "
               "9 variables) se desempata por la PD de CatBoost, luego por el puntaje de Isolation Forest (el perfil más raro primero) y al final por el N.º de cliente. "
               "'Empata con' dice cuántos clientes de toda la base comparten esa PD exacta.")
    st.download_button("Descargar esta lista (CSV)", t2.to_csv(index=False).encode("utf-8"), "criticos.csv", "text/csv")
    ver = st.selectbox("Ver el detalle de uno de estos clientes", ["—"] + [int(x) for x in top["n_cliente"]])
    if ver != "—":
        panel_cliente(fila_cruda(int(ver)), f"Cliente N.º {int(ver):,}", f"v{int(ver)}", int(ver), int(base.iloc[int(ver) - 1]["id"]))

# ---------------- 3. ingesta de nuevos clientes ----------------
if pest == PESTANAS[2]:
    st.markdown("Se sube un CSV con solicitudes **nuevas con sus valores crudos** (ingreso, DTI, monto, puntaje de buró, antigüedad, destino, vivienda y estado). "
                "Cada fila se transforma con lo aprendido en el entrenamiento (tramos WoE y escalas), sin recalcular nada con el lote.")
    st.markdown("**Probar sin archivo:** un clic genera clientes con todas las variables al azar y los puntúa al instante.")
    g1, g2 = st.columns([1, 2])
    ns = g1.number_input("Cantidad de clientes", 1, 500, 10, 1)
    g2.write("")
    g2.write("")
    if g2.button("Generar clientes aleatorios", type="primary"):
        st.session_state["lote"] = simular_aleatorio(int(ns))
    archivo = st.file_uploader("O subir un CSV propio", type="csv")
    if archivo is not None and st.session_state.get("ultimo_archivo") != archivo.file_id:
        st.session_state["lote"] = pd.read_csv(archivo)
        st.session_state["ultimo_archivo"] = archivo.file_id
    if "lote" not in st.session_state:
        st.session_state["lote"] = simular_aleatorio(10)      # la pestaña abre con 10 clientes ya puntuados
    lote = st.session_state.get("lote")
    if lote is not None:
        problemas = validar(lote)
        if problemas:
            st.error("No se puede puntuar el archivo: " + "; ".join(problemas))
        else:
            col_id = lote.columns[0] if lote.columns[0] not in COLUMNAS_CRUDAS else None
            ids = lote[col_id].astype(str).to_numpy() if col_id else np.arange(1, len(lote) + 1).astype(str)
            res, tratado_l, woe_l = puntuar(lote.reset_index(drop=True), cfg, art, CUT)
            res = res.reset_index(drop=True)
            res["id"] = ids
            conteo = res["decision"].value_counts()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Solicitudes", f"{len(res):,}")
            k2.metric("Crédito aprobado", f"{conteo.get('Crédito aprobado', 0):,}")
            k3.metric("Crédito no aprobado", f"{conteo.get('Crédito no aprobado', 0):,}")
            cerca = int(((res["pd_xgb"] < CUT) & (res["pd_xgb"] >= 0.8 * CUT)).sum())
            k4.metric("Aprobadas cerca del umbral", f"{cerca:,}", help="Entre el 80 % y el 100 % del cut-off")
            res = res.sort_values(["pd_xgb", "pd_cat", "score_if"], ascending=False)
            t3 = tabla_lote(res, ("ID", res["id"].to_numpy()), CUT)
            t3.insert(1, "Decisión", np.where(res["pd_xgb"].to_numpy() >= CUT, "No aprobado",
                                              np.where(res["pd_xgb"].to_numpy() >= 0.8 * CUT, "Aprobado, cerca del umbral", "Aprobado")))
            t3["DTI fuera de soporte"] = np.where(res["fuera_soporte_dti"].to_numpy(), "Sí", "")
            if "nota_simulacion" in lote.columns:
                t3["Nota del simulador"] = lote.reset_index(drop=True).loc[res.index, "nota_simulacion"].to_numpy()
            st.dataframe(t3, hide_index=True, width="stretch", column_config=CONFIG_COLS, height=min(700, 38 * (len(t3) + 1)))
            st.caption("Ordenado de mayor a menor PD de XGBoost. Las filas con 'Atípico' en algún detector merecen revisión aunque los modelos aprueben.")
            st.download_button("Descargar los clientes ingresados (CSV con valores crudos)", lote.to_csv(index=False).encode("utf-8"),
                               "clientes_ingresados.csv", "text/csv")
            st.download_button("Descargar resultados (CSV)", t3.to_csv(index=False).encode("utf-8"), "resultados_nuevos_clientes.csv", "text/csv")
            with st.expander("Ver cómo se transformó una solicitud (dato crudo → tratado → WoE)"):
                elegido = st.selectbox("Solicitud", list(ids))
                i = list(ids).index(elegido)
                crudo_i = lote.reset_index(drop=True).loc[i, COLUMNAS_CRUDAS]
                tr_i = tratado_l.reset_index(drop=True).loc[i]
                woe_i = woe_l.reset_index(drop=True).loc[i]
                st.dataframe(pd.DataFrame({"Dato crudo": crudo_i.astype(str).to_numpy()}, index=COLUMNAS_CRUDAS).T, width="stretch")
                tabla = tabla_tramos(tr_i, woe_i)[["Variable", "Valor", "Tramo", "WoE"]]
                st.dataframe(tabla, hide_index=True, width="stretch")
                st.caption("El WoE de cada tramo se calculó con el entrenamiento. Ese valor es lo que recibe el modelo.")
    else:
        st.info("Para ver el análisis hay que subir un archivo o generar solicitudes simuladas.")
        st.download_button("Descargar un CSV de ejemplo (10 solicitudes)", (RAIZ / "data/ejemplos/nuevos_clientes_10.csv").read_bytes(),
                           "nuevos_clientes_10.csv", "text/csv")

# ---------------- 4. política ----------------
if pest == PESTANAS[3]:
    ref = referencia_validacion()
    st.markdown("Efecto del cut-off sobre las **123,633 solicitudes de validación** (2014T3 a 2014T4), donde sí sabemos quién pagó. "
                "Se usan el margen y la LGD de la barra lateral.")

    def resumen(m_, l_, cut_):
        aprobar = ref["pd_calibrada"] < cut_
        y_, monto = ref["default"], ref["loan_amnt"]
        ganancia = (((1 - y_) * m_ - y_ * l_) * monto)[aprobar].sum()
        return {"aprobar": aprobar.mean(), "default_aprobados": y_[aprobar].mean(), "malos_aprobados": int(y_[aprobar].sum()), "ganancia": ganancia}

    r = resumen(margen, lgd, CUT)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Cut-off", f"{CUT:.2%}")
    a2.metric("Se aprueba", f"{r['aprobar']:.1%}")
    a3.metric("Default real de aprobados", f"{r['default_aprobados']:.1%}", help=f"En toda la validación el default es {ref['default'].mean():.1%}.")
    a4.metric("Ganancia de aprobados (USD)", f"{r['ganancia'] / 1e6:,.1f} M")
    st.caption(f"Default de toda la validación: {ref['default'].mean():.1%}. Malos aprobados (falsos negativos): {r['malos_aprobados']:,}. "
               "La ganancia usa el monto real y si cada cliente pagó o no: margen sobre los que pagan menos LGD sobre los que no.")
    cortes = np.linspace(0.03, 0.30, 55)
    y, monto, pdv = ref["default"].values, ref["loan_amnt"].values, ref["pd_calibrada"].values
    ganancias = [(((1 - y) * margen - y * lgd) * monto)[pdv < c].sum() / 1e6 for c in cortes]
    fig = go.Figure(go.Scatter(x=cortes * 100, y=ganancias, mode="lines", line=dict(color=AZUL, width=3)))
    fig.add_vline(x=CUT * 100, line_dash="dash", line_color=ROJO, annotation_text=f"cut-off {CUT:.1%}")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=30), xaxis_title="Aprobar si la PD es menor que (%)", yaxis_title="Ganancia (USD millones)")
    st.plotly_chart(fig, key="pol_curva")
    st.caption("Si la curva tiene su máximo cerca del corte de costo, la PD está bien calibrada.")
    st.markdown("**Grilla de sensibilidad**")
    filas = []
    for m_ in dec["grilla_margen"]:
        for l_ in dec["grilla_lgd"]:
            c_ = corte_costo(m_, l_)
            x = resumen(m_, l_, c_)
            filas.append({"Margen": f"{m_:.0%}", "LGD": f"{l_:.0%}", "Corte": f"{c_:.1%}", "Aprobar": f"{x['aprobar']:.1%}",
                          "Default aprobados": f"{x['default_aprobados']:.1%}", "Malos aprobados": f"{x['malos_aprobados']:,}",
                          "Ganancia (M USD)": round(x["ganancia"] / 1e6, 1)})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

# ---------------- 5. modelo ----------------
if pest == PESTANAS[4]:
    mt = ref_c["metricas"]
    st.markdown(f"""
**Qué es.** Un score de originación: estima, al momento de la solicitud, la probabilidad de que el cliente no pague (PD). Se entrenó con
préstamos de LendingClub emitidos entre 2007T2 y 2014T2 (330,171 créditos, 16.2 % de default) y se probó en dos periodos que los modelos nunca vieron.

**Cómo decide.** Dos modelos sobre 9 variables transformadas a WoE: **XGBoost** ({art['modelo_final']['arboles']} árboles, el campeón) y **CatBoost** (el retador),
los dos con calibración sigmoide para leer la salida como probabilidad. La decisión usa la PD de XGBoost frente al cut-off; CatBoost sirve de segunda opinión.
El CatBoost no se reentrenó: se le ajustó la misma calibración con las predicciones de validación (umbral F1 propio: {ref_c['umbral_cat']:.1%}).

| Medida | XGBoost validación | CatBoost validación | XGBoost fuera de tiempo | CatBoost fuera de tiempo |
|---|---|---|---|---|
| Solicitudes | {mt['xgb']['validacion']['n']:,} | {mt['cat']['validacion']['n']:,} | {mt['xgb']['oot']['n']:,} | {mt['cat']['oot']['n']:,} |
| Tasa de default | {mt['xgb']['validacion']['tasa']:.1%} | {mt['cat']['validacion']['tasa']:.1%} | {mt['xgb']['oot']['tasa']:.1%} | {mt['cat']['oot']['tasa']:.1%} |
| AUC | {mt['xgb']['validacion']['auc']:.4f} | {mt['cat']['validacion']['auc']:.4f} | {mt['xgb']['oot']['auc']:.4f} | {mt['cat']['oot']['auc']:.4f} |
| Brier | {mt['xgb']['validacion']['brier']:.4f} | {mt['cat']['validacion']['brier']:.4f} | {mt['xgb']['oot']['brier']:.4f} | {mt['cat']['oot']['brier']:.4f} |
| PD media | {mt['xgb']['validacion']['pd_media']:.1%} | {mt['cat']['validacion']['pd_media']:.1%} | {mt['xgb']['oot']['pd_media']:.1%} | {mt['cat']['oot']['pd_media']:.1%} |

**Cómo entra un cliente nuevo.** Con sus valores crudos. Se limpia con los parámetros del entrenamiento (ingreso entre 18,200 y 240,000, DTI inválido a la mediana 16.42,
categorías raras o nuevas a "otros"), se calcula préstamo/ingreso y cada dato se ubica en el tramo WoE del entrenamiento. Para los detectores se escala a 0–1 con el mínimo y el máximo del
entrenamiento. Nada se recalcula con el lote que llega.

**Los tres detectores.** Isolation Forest, GMM y ECOD no predicen la PD: catalogan el perfil como Normal, Inusual o Atípico frente a los clientes de entrenamiento
(percentiles 90 y 99). Son un contraste sugerible, no cambian la decisión.

**Límites.** El AUC cercano a 0.66 ordena bien pero no separa del todo a buenos y malos. Los datos son de EE. UU.: antes de usarlo en una entidad ecuatoriana hay que
recalibrarlo con su propia cartera. El margen y la LGD son supuestos que define el comité.
""")
