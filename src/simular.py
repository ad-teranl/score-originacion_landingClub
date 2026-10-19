"""Simulador de solicitudes nuevas (valores crudos) para probar la ingesta por CSV.

Cada solicitud parte de un cliente real de la base, tomado al azar con sus 8 datos juntos (así se conservan las
relaciones entre variables), y se le agrega un ruido aleatorio pequeño. Un 10 % sale "raro" a propósito
(DTI alto, ingreso extremo, monto grande para el ingreso o categoría nunca vista) para que los detectores tengan qué marcar.
Con la misma semilla sale el mismo archivo.
"""
import numpy as np
import pandas as pd

COLUMNAS = ["revenue", "dti_n", "loan_amnt", "fico_n", "emp_length", "purpose", "home_ownership_n", "addr_state"]
RAROS = ["DTI muy alto", "Ingreso extremo", "Monto alto para su ingreso", "Categoría nunca vista", "DTI sin dato"]


def simular(base, n=10, semilla=42, frac_raros=0.10, prefijo="SOL"):
    rng = np.random.default_rng(semilla)
    s = base.iloc[rng.choice(len(base), size=n, replace=False)][COLUMNAS].reset_index(drop=True).copy()
    for c in ["emp_length", "purpose", "home_ownership_n", "addr_state"]:
        s[c] = s[c].astype(str)
    s["revenue"] = (s["revenue"] * rng.lognormal(0, 0.10, n)).round(-2).clip(lower=6000)
    s["loan_amnt"] = ((s["loan_amnt"] * rng.uniform(0.85, 1.15, n)) / 25).round() * 25
    s["loan_amnt"] = s["loan_amnt"].clip(1000, 40000).astype(int)
    s["dti_n"] = (s["dti_n"] + rng.normal(0, 2, n)).clip(lower=0).round(2)
    s["fico_n"] = (s["fico_n"] + rng.normal(0, 5, n)).round().clip(600, 850)
    s["nota_simulacion"] = "normal"
    n_raros = max(1, int(round(n * frac_raros))) if frac_raros > 0 and n >= 5 else 0
    for k, i in enumerate(rng.choice(n, size=n_raros, replace=False)):
        tipo = RAROS[k % len(RAROS)]
        if tipo == "DTI muy alto":
            s.loc[i, "dti_n"] = round(float(rng.uniform(36, 60)), 2)
        elif tipo == "Ingreso extremo":
            s.loc[i, "revenue"] = float(rng.integers(300, 700) * 1000)
        elif tipo == "Monto alto para su ingreso":
            s.loc[i, "revenue"] = float(rng.integers(20, 30) * 1000)
            s.loc[i, "loan_amnt"] = int(rng.integers(30, 40) * 1000)
        elif tipo == "Categoría nunca vista":
            s.loc[i, "purpose"] = "boat"
            s.loc[i, "addr_state"] = "ZZ"
        else:
            s.loc[i, "dti_n"] = np.nan
        s.loc[i, "nota_simulacion"] = "raro: " + tipo.lower()
    s.insert(0, "id_solicitud", [f"{prefijo}-{i + 1:04d}" for i in range(n)])
    return s


def simular_aleatorio(n=10, semilla=None, prefijo="SOL"):
    """Solicitudes 100 % aleatorias: cada variable se sortea por separado dentro de rangos plausibles.

    Sin semilla, cada llamada produce clientes distintos. Como las variables no se relacionan entre sí,
    pueden salir combinaciones poco realistas (por ejemplo ingreso bajo con monto alto): sirve para probar el tablero."""
    rng = np.random.default_rng(semilla)
    purpose = ["car", "credit_card", "debt_consolidation", "home_improvement", "house", "major_purchase",
               "medical", "moving", "other", "small_business", "vacation", "wedding"]
    emp = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]
    estados = ["AL", "AR", "AZ", "CA", "CO", "CT", "FL", "GA", "HI", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "MI", "MN", "MO",
               "NC", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "SC", "TN", "TX", "UT", "VA", "WA", "WI", "WV"]
    d = pd.DataFrame({
        "id_solicitud": [f"{prefijo}-{i + 1:02d}" for i in range(n)],
        "revenue": np.round(rng.lognormal(np.log(60000), 0.55, n), -2).clip(12000, 300000),
        "dti_n": np.round(rng.uniform(1, 38, n), 2),
        "loan_amnt": (rng.integers(20, 801, n) * 50),
        "fico_n": (rng.integers(132, 171, n) * 5).astype(float),
        "emp_length": rng.choice(emp, n),
        "purpose": rng.choice(purpose, n),
        "home_ownership_n": rng.choice(["RENT", "MORTGAGE", "OWN"], n),
        "addr_state": rng.choice(estados, n),
    })
    return d
