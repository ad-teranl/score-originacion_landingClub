"""Control de la puesta en marcha: confirma que la cadena reproduce lo que dejaron los cuadernos.

Uso:
  python verificar.py                      # rapido: 500 clientes de la base se vuelven a puntuar desde sus datos crudos
  python verificar.py --completo RUTA      # ademas puntua toda la validacion (RUTA = carpeta SCORE_ORIGINACION)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from puntuar import COLUMNAS_CRUDAS, cargar_artefactos, cargar_config, puntuar  # noqa: E402

RAIZ = Path(__file__).resolve().parent


def control_base(cfg, art):
    base = pd.read_parquet(RAIZ / cfg["rutas"]["modelos"] / cfg["base"]["archivo"])
    muestra = base.sample(500, random_state=7)
    crudo = muestra[COLUMNAS_CRUDAS].copy()
    for c in ["emp_length", "purpose", "home_ownership_n", "addr_state"]:
        crudo[c] = crudo[c].astype(str)
    salida, _, _ = puntuar(crudo, cfg, art)
    d_x = np.abs(salida["pd_xgb"] - muestra["pd_xgb"]).max()
    d_c = np.abs(salida["pd_cat"] - muestra["pd_cat"]).max()
    d_i = np.abs(salida["score_if"] - muestra["score_if"]).max()
    print(f"Base ({len(base):,} clientes), 500 re-puntuados | dif. PD XGBoost = {d_x:.1e} | PD CatBoost = {d_c:.1e} | IF = {d_i:.1e}")
    ref = art["ref_cartera"]["metricas"]
    auc = ref["xgb"]["validacion"]["auc"]
    print(f"AUC guardado: XGBoost validación = {auc:.5f} (cuaderno 4: {cfg['referencia']['auc_validacion']:.5f}) | "
          f"CatBoost = {ref['cat']['validacion']['auc']:.5f}")
    return d_x < 1e-6 and d_c < 1e-6 and d_i < 1e-4 and abs(auc - cfg["referencia"]["auc_validacion"]) < 1e-4


def control_completo(cfg, art, ruta):
    from sklearn.metrics import roc_auc_score
    ruta = Path(ruta)
    eda = pd.read_parquet(ruta / "datos/lc_eda.parquet")
    valid = pd.read_parquet(ruta / "datos/valid_tratado.parquet")
    salida, _, _ = puntuar(eda.loc[valid.index, COLUMNAS_CRUDAS], cfg, art)
    ref = cfg["referencia"]
    ok = True
    for m, esperado in [("pd_xgb", ref["auc_validacion"]), ("pd_cat", None)]:
        auc = roc_auc_score(valid["Default"], salida[m])
        print(f"Validación {m}: {len(valid):,} filas | AUC = {auc:.5f} | PD media = {salida[m].mean():.4f}"
              + (f" (cuaderno 4: {esperado:.5f})" if esperado else ""))
        if esperado:
            ok = ok and abs(auc - esperado) < 1e-6
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--completo", default=None)
    args = p.parse_args()
    cfg = cargar_config()
    art = cargar_artefactos(cfg)
    ok = control_base(cfg, art)
    if args.completo:
        ok = control_completo(cfg, art, args.completo) and ok
    print("RESULTADO:", "OK, la cadena reproduce los cuadernos" if ok else "REVISAR, hay diferencias")
    sys.exit(0 if ok else 1)
