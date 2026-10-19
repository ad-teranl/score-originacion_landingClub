"""Puntúa la base completa (todas las solicitudes del archivo original) y guarda las tablas de referencia del tablero.

No reentrena nada. Usa los .pkl de models/ y el parquet de datos (lc_eda.parquet, salida del cuaderno 1).
Uso:  python scripts/construir_base.py "RUTA\\SCORE_ORIGINACION"
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from cadena_pd import COLUMNAS_CRUDAS  # noqa: E402
from modelado_pd import CalibradorPD, _logit  # noqa: E402
from puntuar import cargar_artefactos, cargar_config  # noqa: E402

ruta = Path(sys.argv[1])
cfg = cargar_config()
art = cargar_artefactos(cfg)
M = RAIZ / "models"

eda = pd.read_parquet(ruta / "datos" / "lc_eda.parquet")
print("filas:", len(eda))

# ---------- 1. CatBoost: calibrador sigmoide y umbral F1 (misma lógica que el ganador) ----------
cat = joblib.load(M / "modelo_CatBoost_E3.pkl")
pred = joblib.load(ruta / "salidas_4_SUPERVISADOS" / "modelos" / "pred_CatBoost_E3.pkl")
# pred["valid"] viene en el orden de valid_tratado (el del cuaderno 4), no en el orden de lc_eda
valid_tratado = pd.read_parquet(ruta / "datos" / "valid_tratado.parquet", columns=["Default"])
y_valid = valid_tratado["Default"].to_numpy()
p_valid = np.asarray(pred["valid"], dtype=float)
cal_cat = CalibradorPD("sigmoide").fit(p_valid, y_valid)
oof = np.zeros(len(p_valid))
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(p_valid, y_valid):
    oof[b] = CalibradorPD("sigmoide").fit(p_valid[a], y_valid[a]).predict(p_valid[b])
malla = np.round(np.arange(0.01, 0.601, 0.005), 3)
f1s = [f1_score(y_valid, oof >= u) for u in malla]
umbral_cat = float(malla[int(np.argmax(f1s))])
print(f"CatBoost: AUC valid = {roc_auc_score(y_valid, p_valid):.5f} | umbral F1 = {umbral_cat} | F1 = {max(f1s):.4f}")
joblib.dump({"calibrador": cal_cat, "umbral": umbral_cat, "columnas": art["modelo_final"]["columnas"]}, M / "calibrador_CatBoost.pkl")

# ---------- 2. puntuar toda la base ----------
crudo = eda[COLUMNAS_CRUDAS]
woe = art_woe = joblib.load(M / "cadena_E3.pkl").transform(crudo)
cols = art["modelo_final"]["columnas"]
xgb_raw = art["modelo_final"]["modelo"].predict_proba(woe[cols])[:, 1]
pd_xgb = art["modelo_final"]["calibrador"].predict(xgb_raw)
cat_raw = cat.predict_proba(woe[cols])[:, 1]
pd_cat = cal_cat.predict(cat_raw)
X = joblib.load(M / "cadena_nosup_E3.pkl").transform(crudo)
s_if = -art["isolation_forest"].score_samples(X[list(art["isolation_forest"].feature_names_in_)])
s_gmm = -art["gmm"].score_samples(X[list(art["gmm"].feature_names_in_)])
s_ecod = art["ecod"]["modelo"].score_samples(X[art["ecod"]["columnas"]].values)
seg = art["gmm"].predict(X[list(art["gmm"].feature_names_in_)])

# control: AUC de validación igual al cuaderno 4
mv = (eda["bloque"] == "validacion").to_numpy()
print(f"XGBoost AUC valid = {roc_auc_score(eda.loc[mv, 'Default'], pd_xgb[mv]):.5f} | CatBoost = {roc_auc_score(eda.loc[mv, 'Default'], pd_cat[mv]):.5f}")

base = pd.DataFrame({
    "n_cliente": np.arange(1, len(eda) + 1, dtype=np.int32),
    "id": pd.to_numeric(eda.index).astype("int64"),
}).reset_index(drop=True)
for c in ["revenue", "dti_n", "loan_amnt", "fico_n"]:
    base[c] = eda[c].to_numpy()
for c in ["emp_length", "purpose", "home_ownership_n", "addr_state"]:
    base[c] = eda[c].astype(str).to_numpy()
    base[c] = base[c].astype("category")
base["pd_xgb"] = pd_xgb
base["pd_cat"] = pd_cat
base["score_if"] = s_if.astype(np.float32)
base["score_gmm"] = s_gmm.astype(np.float32)
base["score_ecod"] = s_ecod.astype(np.float32)
base["seg_gmm"] = seg.astype(np.int8)
# en 3 partes (<30 MB cada una) para que GitHub y las copias de archivos las manejen sin problema
carpeta_base = M / "base_clientes"
carpeta_base.mkdir(exist_ok=True)
cortes = np.linspace(0, len(base), 4).astype(int)
for i in range(3):
    base.iloc[cortes[i]:cortes[i + 1]].to_parquet(carpeta_base / f"parte_{i + 1}.parquet", index=False, compression="zstd")

# ---------- 3. referencias ----------
tr = (eda["bloque"] == "entrenamiento").to_numpy()
ref = {"n_base": int(len(eda)), "n_entrenamiento": int(tr.sum())}
nosup = {}
for nombre, s in [("if", s_if), ("gmm", s_gmm), ("ecod", s_ecod)]:
    p90, p99 = np.percentile(s[tr], [90, 99])
    lo, hi = np.percentile(s[tr], [0.2, 99.9])
    cuentas, bordes = np.histogram(np.clip(s[tr], lo, hi), bins=60)
    nosup[nombre] = {"p90": float(p90), "p99": float(p99), "bordes": bordes.tolist(), "cuentas": cuentas.tolist()}
print({k: (round(v["p90"], 4), round(v["p99"], 4)) for k, v in nosup.items()})
y = eda["Default"].to_numpy()
segs = pd.DataFrame({"seg": seg[tr], "y": y[tr]}).groupby("seg")["y"].agg(["mean", "count"])
nosup["segmentos_gmm"] = {int(k): {"tasa_default": float(r["mean"]), "n": int(r["count"])} for k, r in segs.iterrows()}
nosup["componentes"] = int(art["gmm"].n_components)
json.dump(nosup, open(M / "referencia_nosup.json", "w"))

def hist(p):
    cuentas, bordes = np.histogram(p, bins=80, range=(0, min(1.0, float(np.percentile(p, 99.9)) * 1.05)))
    return {"bordes": bordes.tolist(), "cuentas": cuentas.tolist()}

cuant = {}
for c in ["revenue", "dti_n", "loan_amnt", "fico_n"]:
    cuant[c] = np.percentile(eda[c].dropna(), np.arange(0, 101)).tolist()
cuant["loan_to_income"] = np.percentile((eda["loan_amnt"] / eda["revenue"].clip(18200, 240000)).dropna(), np.arange(0, 101)).tolist()
# métricas por bloque (validación y fuera de tiempo) de los dos modelos
from sklearn.metrics import brier_score_loss
met = {}
for bl, nombre in [("validacion", "validacion"), ("oot", "oot")]:
    m_ = (eda["bloque"] == bl).to_numpy()
    for modelo, p in [("xgb", pd_xgb), ("cat", pd_cat)]:
        met.setdefault(modelo, {})[nombre] = {"n": int(m_.sum()), "tasa": float(y[m_].mean()), "auc": float(roc_auc_score(y[m_], p[m_])),
                                              "brier": float(brier_score_loss(y[m_], p[m_])), "pd_media": float(p[m_].mean())}
ref["metricas"] = met
ref["umbral_cat"] = umbral_cat
ref.update({"hist_xgb": hist(pd_xgb), "hist_cat": hist(pd_cat), "cuantiles": cuant,
            "pd_media_xgb": float(pd_xgb.mean()), "pd_media_cat": float(pd_cat.mean()),
            "tasa_default_entrenamiento": float(y[tr].mean()),
            "deciles_xgb": np.percentile(pd_xgb, np.arange(10, 100, 10)).tolist()})
json.dump(ref, open(M / "referencia_cartera.json", "w"))

# tabla WoE (tramos y tasa de default del entrenamiento) para explicar cada dato
art["pipeline_woe"].named_steps["woe"].tabla_woe_.to_parquet(M / "tabla_woe.parquet", index=False)
print("listo")
