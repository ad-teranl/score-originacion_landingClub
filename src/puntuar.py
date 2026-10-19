"""Núcleo del score de originación: de la solicitud cruda a la decisión. Nada se reentrena.

Cadena (cada pieza se ajustó una vez con entrenamiento y aquí solo se aplica transform):
  cadena_E3.pkl        limpieza (cuaderno 2) -> préstamo/ingreso + WoE (cuaderno 3)    -> 9 WoE
  XGBoost + calibrador (cuaderno 4) y CatBoost + calibrador                            -> 2 PD
  cadena_nosup_E3.pkl  limpieza -> préstamo/ingreso -> MinMax compacta (cuaderno 3)    -> 9 columnas
  Isolation Forest, GMM, ECOD (cuaderno 5)                                              -> 3 categorías
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
# Los .pkl guardan referencias a estos módulos: deben poder importarse.
sys.path.insert(0, str(RAIZ / "src"))
from cadena_pd import COLUMNAS_CRUDAS  # noqa: E402,F401

NOMBRES_ES = {
    "woe_revenue_w": "Ingreso anual", "woe_dti_n_imp": "Deuda / ingreso (DTI)", "woe_loan_amnt": "Monto solicitado",
    "woe_fico_n": "Puntaje de buró", "woe_loan_to_income": "Préstamo / ingreso", "woe_purpose_g": "Destino del crédito",
    "woe_home_ownership_n_g": "Vivienda", "woe_addr_state_g": "Estado", "woe_emp_length": "Antigüedad laboral",
}
VARIABLE_DE_WOE = {c: c[4:] for c in NOMBRES_ES}   # woe_revenue_w -> revenue_w


OMITIR = {"pipeline_woe", "objetos_ingenieria", "logistica"}


def cargar_config(ruta=None):
    with open(ruta or RAIZ / "config" / "main.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_artefactos(cfg):
    carpeta = RAIZ / cfg["rutas"]["modelos"]
    a = cfg["artefactos"]
    # pipeline_woe, objetos_ingenieria y logistica solo sirven para reconstruir las cadenas: no se cargan (ahorra memoria)
    art = {k: joblib.load(carpeta / v) for k, v in a.items() if v.endswith(".pkl") and k not in OMITIR}
    art["tarjeta_puntos"] = pd.read_parquet(carpeta / a["tarjeta_puntos"])
    art["tabla_woe"] = pd.read_parquet(carpeta / a["tabla_woe"])
    art["ref_nosup"] = json.load(open(carpeta / a["referencia_nosup"], encoding="utf-8"))
    art["ref_cartera"] = json.load(open(carpeta / a["referencia_cartera"], encoding="utf-8"))
    return art


# ---------- entrada ----------
def validar(crudo):
    """Lista de problemas que impiden puntuar (columnas faltantes o valores imposibles)."""
    problemas = [f"falta la columna {c}" for c in COLUMNAS_CRUDAS if c not in crudo.columns]
    if problemas:
        return problemas
    for c in ["revenue", "loan_amnt", "fico_n"]:
        v = pd.to_numeric(crudo[c], errors="coerce")
        malas = v.isna() | (v <= 0)
        if malas.any():
            filas = ", ".join(str(i + 2) for i in np.flatnonzero(malas.to_numpy())[:8])
            problemas.append(f"{c}: debe ser un número mayor que cero (revisar filas {filas}{'…' if malas.sum() > 8 else ''} del archivo)")
    return problemas


# ---------- cut-off ----------
def corte_costo(margen, lgd):
    """Se aprueba si la ganancia esperada es positiva: (1-PD)*margen > PD*LGD  =>  PD < margen/(margen+LGD)."""
    return margen / (margen + lgd)


def cut_off(cfg, modo="f1", margen=None, lgd=None):
    d = cfg["decision"]
    if modo == "costo":
        return corte_costo(margen or d["margen"], lgd or d["lgd"])
    return float(d["umbral_f1"])


# ---------- categorías de los no supervisados ----------
def categoria(score, p90, p99):
    return np.where(score > p99, "Atípico", np.where(score > p90, "Inusual", "Normal"))


def categorias_nosup(scores, art):
    r = art["ref_nosup"]
    return {n: categoria(scores[n], r[n]["p90"], r[n]["p99"]) for n in ["if", "gmm", "ecod"]}


# ---------- explicación ----------
def contribuciones_xgb(woe, art):
    """SHAP exacto de XGBoost (pred_contribs), en log-odds. Positivo = sube el riesgo."""
    import xgboost as xgb
    m = art["modelo_final"]
    c = m["modelo"].get_booster().predict(xgb.DMatrix(woe[m["columnas"]]), pred_contribs=True)
    return pd.DataFrame(c[:, :-1], index=woe.index, columns=m["columnas"]), float(c[0, -1])


def contribuciones_cat(woe, art):
    """SHAP nativo de CatBoost (ShapValues), en log-odds."""
    from catboost import Pool
    cols = art["modelo_final"]["columnas"]
    c = art["catboost"].get_feature_importance(Pool(woe[cols]), type="ShapValues")
    return pd.DataFrame(c[:, :-1], index=woe.index, columns=cols), float(c[0, -1])


contribuciones = contribuciones_xgb


def puntos_tarjeta(woe, art, detalle=False):
    """Puntos de la tarjeta (logística E3): suma de los puntos del tramo de cada variable."""
    tarjeta = art["tarjeta_puntos"]
    total = pd.Series(0.0, index=woe.index)
    partes = {}
    for var, grupo in tarjeta.groupby("variable", sort=False):
        mapa = dict(zip(grupo["woe"].round(8), grupo["puntos"]))
        p = woe["woe_" + var].round(8).map(mapa).fillna(grupo["puntos"].median())
        partes["woe_" + var] = p
        total += p
    return (total, pd.DataFrame(partes)) if detalle else total


# ---------- puntuación ----------
def puntuar(crudo, cfg=None, art=None, cutoff=None):
    """Puntúa solicitudes crudas. Devuelve (salida, tratado, woe); salida tiene una fila por solicitud."""
    cfg = cfg or cargar_config()
    art = art or cargar_artefactos(cfg)
    problemas = validar(crudo)
    if problemas:
        raise ValueError("; ".join(problemas))
    cutoff = cut_off(cfg) if cutoff is None else cutoff
    x = crudo[COLUMNAS_CRUDAS]
    woe = art["cadena"].transform(x)
    tratado = art["cadena"].named_steps["razon"].transform(art["cadena"].named_steps["tratar"].transform(x))
    cols = art["modelo_final"]["columnas"]
    m = art["modelo_final"]
    pd_xgb = m["calibrador"].predict(m["modelo"].predict_proba(woe[cols])[:, 1])
    pd_cat = art["calibrador_cat"]["calibrador"].predict(art["catboost"].predict_proba(woe[cols])[:, 1])
    Xn = art["cadena_nosup"].transform(x)
    s = {
        "if": -art["isolation_forest"].score_samples(Xn[list(art["isolation_forest"].feature_names_in_)]),
        "gmm": -art["gmm"].score_samples(Xn[list(art["gmm"].feature_names_in_)]),
        "ecod": art["ecod"]["modelo"].score_samples(Xn[art["ecod"]["columnas"]].values),
    }
    seg = art["gmm"].predict(Xn[list(art["gmm"].feature_names_in_)])
    cat = categorias_nosup(s, art)
    dti = pd.to_numeric(crudo["dti_n"], errors="coerce")

    out = pd.DataFrame(index=crudo.index)
    out["pd_xgb"] = pd_xgb
    out["pd_cat"] = pd_cat
    out["puntos_tarjeta"] = puntos_tarjeta(woe, art).round(0)
    out["score_if"], out["score_gmm"], out["score_ecod"] = s["if"], s["gmm"], s["ecod"]
    out["cat_if"], out["cat_gmm"], out["cat_ecod"] = cat["if"], cat["gmm"], cat["ecod"]
    out["segmento_gmm"] = seg
    out["fuera_soporte_dti"] = (dti > cfg["gobernanza"]["dti_max_entrenamiento"]).to_numpy()
    out["cutoff"] = cutoff
    out["aprobado"] = out["pd_xgb"] < cutoff
    out["modelos_coinciden"] = (out["pd_xgb"] < cutoff) == (out["pd_cat"] < cutoff)
    out["decision"] = np.where(out["aprobado"], "Crédito aprobado", "Crédito no aprobado")
    return out, tratado, woe


def detectores_marcan(fila):
    """Cuántos de los 3 detectores catalogan la solicitud como Atípica (p99)."""
    return int(sum(fila[c] == "Atípico" for c in ["cat_if", "cat_gmm", "cat_ecod"]))
