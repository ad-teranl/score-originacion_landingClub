"""Cadena de transformación del dato crudo, armada con lo aprendido en el entrenamiento.

Cada pieza sigue la regla de sklearn: se ajusta una vez con entrenamiento (fit) y a todo lo demás
(validación, fuera de tiempo, cliente nuevo) solo se le aplica transform(). Nada se recalcula con el lote.

  cadena_E3.pkl         limpieza -> préstamo/ingreso -> WoE   (entrada de XGBoost y CatBoost, 9 columnas)
  cadena_nosup_E3.pkl   limpieza -> préstamo/ingreso -> compacta MinMax   (entrada de IF, GMM y ECOD, 9 columnas)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

COLUMNAS_CRUDAS = ["revenue", "dti_n", "loan_amnt", "fico_n", "emp_length",
                   "purpose", "home_ownership_n", "addr_state"]


class TratadorCrudo(BaseEstimator, TransformerMixin):
    """Cuaderno 2: dato crudo -> dato tratado. Los parámetros se fijaron con el entrenamiento."""

    def __init__(self, revenue_min=18200.0, revenue_max=240000.0, dti_centinela=100.0, dti_mediana=16.42,
                 etiqueta_rara="otros", niveles_validos=None, emp_niveles=None):
        self.revenue_min = revenue_min
        self.revenue_max = revenue_max
        self.dti_centinela = dti_centinela
        self.dti_mediana = dti_mediana
        self.etiqueta_rara = etiqueta_rara
        self.niveles_validos = niveles_validos
        self.emp_niveles = emp_niveles

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        d = pd.DataFrame(index=X.index)
        d["revenue_w"] = pd.to_numeric(X["revenue"]).clip(self.revenue_min, self.revenue_max)
        dti = pd.to_numeric(X["dti_n"], errors="coerce")
        d["dti_n_imp"] = dti.where(dti <= self.dti_centinela).fillna(self.dti_mediana)
        d["loan_amnt"] = pd.to_numeric(X["loan_amnt"])
        d["fico_n"] = pd.to_numeric(X["fico_n"]).astype(float)
        for col in ["purpose", "home_ownership_n", "addr_state"]:
            v = X[col].astype(str).str.strip()
            d[col + "_g"] = v.where(v.isin(self.niveles_validos[col]), self.etiqueta_rara)
        emp = X["emp_length"].fillna("NI").astype(str).str.strip()
        d["emp_length"] = emp.where(emp.isin(self.emp_niveles), "NI")
        return d


class CompactaNoSup(BaseEstimator, TransformerMixin):
    """Cuaderno 3: matriz compacta de 9 columnas con MinMax del entrenamiento (entrada de los no supervisados)."""

    def __init__(self, frecuencias=None, escalador=None):
        self.frecuencias = frecuencias
        self.escalador = escalador

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        c = pd.DataFrame(index=X.index)
        c["revenue_w_log"] = np.log1p(X["revenue_w"])
        c["dti_n_imp"] = X["dti_n_imp"]
        c["loan_amnt"] = X["loan_amnt"]
        c["fico_n"] = X["fico_n"]
        c["loan_to_income"] = X["loan_to_income"]
        for col, nombre in [("purpose_g", "purpose"), ("home_ownership_n_g", "home_ownership_n"),
                            ("addr_state_g", "addr_state"), ("emp_length", "emp_length")]:
            c[nombre + "_frec"] = X[col].map(self.frecuencias[col]).fillna(0.0).astype(float)
        cols = list(self.escalador.feature_names_in_)
        return pd.DataFrame(self.escalador.transform(c[cols]), index=c.index, columns=[x + "_mm" for x in cols])
