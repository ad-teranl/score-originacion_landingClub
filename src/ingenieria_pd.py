"""Clases del cuaderno 3 (ingenieria de variables): razon prestamo/ingreso y transformador WoE.

Se guardan en un archivo aparte porque los objetos que se guardan con joblib (.pkl)
necesitan poder importar sus clases al cargarse en otro cuaderno.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin


def etiquetas_intervalos(cortes):
    """Texto de cada intervalo a partir de los cortes internos: (-inf, a], (a, b], ..., (z, inf)."""
    limites = [-np.inf] + [float(c) for c in cortes] + [np.inf]
    return [f"({limites[i]:g}, {limites[i + 1]:g}]" for i in range(len(limites) - 1)]


def unir_intervalos(cortes, n, b, i):
    """Une los intervalos i e i+1: quita el corte que los separa y suma sus conteos."""
    cortes = np.delete(cortes, i)
    n = np.concatenate([n[:i], [n[i] + n[i + 1]], n[i + 2:]])
    b = np.concatenate([b[:i], [b[i] + b[i + 1]], b[i + 2:]])
    return cortes, n, b


def bordes_binning(x, y, n_inicial=20, min_pct=0.05, max_bins=6, monotono=True):
    """Cortes internos de una variable continua.

    Paso 0: cortes de igual frecuencia (n_inicial intervalos).
    Paso A: se unen los intervalos con menos de min_pct de las filas.
    Paso B: se unen los vecinos que rompen la monotonia de la tasa de default.
    Paso C: se unen los vecinos mas parecidos hasta quedar con max_bins o menos.
    Devuelve (cortes, sentido): sentido = +1 si el default sube con la variable, -1 si baja.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    total = len(x)

    cortes = np.unique(np.quantile(x, np.linspace(0, 1, n_inicial + 1)[1:-1]))
    posicion = np.searchsorted(cortes, x, side="left")
    n = np.bincount(posicion, minlength=len(cortes) + 1).astype(float)
    b = np.bincount(posicion, weights=y, minlength=len(cortes) + 1)

    # Paso A: tamano minimo
    while len(n) > 2 and (n / total).min() < min_pct:
        p = int(np.argmin(n))
        tasa = b / np.maximum(n, 1)
        if p == 0:
            i = 0
        elif p == len(n) - 1:
            i = p - 1
        else:
            i = p - 1 if abs(tasa[p] - tasa[p - 1]) <= abs(tasa[p] - tasa[p + 1]) else p
        cortes, n, b = unir_intervalos(cortes, n, b, i)

    # Paso B: monotonia
    rho = spearmanr(x, y)[0]
    sentido = 1 if (np.isnan(rho) or rho >= 0) else -1
    if monotono:
        while len(n) > 2:
            tasa = b / np.maximum(n, 1)
            malos_pares = np.where(sentido * (tasa[1:] - tasa[:-1]) < 0)[0]
            if len(malos_pares) == 0:
                break
            cortes, n, b = unir_intervalos(cortes, n, b, int(malos_pares[0]))

    # Paso C: maximo de intervalos
    while len(n) > max_bins:
        tasa = b / np.maximum(n, 1)
        i = int(np.argmin(np.abs(np.diff(tasa))))
        cortes, n, b = unir_intervalos(cortes, n, b, i)

    return cortes, sentido


def woe_iv(buenos, malos, suavizado=0.5):
    """WoE = ln(% de buenos / % de malos) por intervalo, con suavizado; devuelve (woe, iv_por_intervalo)."""
    k = len(buenos)
    p_buenos = (buenos + suavizado) / (buenos.sum() + suavizado * k)
    p_malos = (malos + suavizado) / (malos.sum() + suavizado * k)
    woe = np.log(p_buenos / p_malos)
    return woe, (p_buenos - p_malos) * woe


class RazonPrestamoIngreso(BaseEstimator, TransformerMixin):
    """Agrega la columna loan_to_income = loan_amnt / revenue_w."""

    def __init__(self, numerador="loan_amnt", denominador="revenue_w", nombre="loan_to_income"):
        self.numerador = numerador
        self.denominador = denominador
        self.nombre = nombre

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.nombre] = X[self.numerador] / X[self.denominador]
        return X


class TransformadorWoE(BaseEstimator, TransformerMixin):
    """Binning + WoE aprendidos con el entrenamiento; transforma cada variable a su WoE.

    continuas  : variables numericas (binning en intervalos monotonos).
    categoricas: variables de texto (un intervalo por nivel).
    Un valor nuevo (nivel desconocido o dato faltante) recibe WoE 0 (neutro).
    """

    def __init__(self, continuas=None, categoricas=None, orden_niveles=None, n_inicial=20,
                 min_pct=0.05, max_bins=6, monotono=True, suavizado=0.5):
        self.continuas = continuas
        self.categoricas = categoricas
        self.orden_niveles = orden_niveles
        self.n_inicial = n_inicial
        self.min_pct = min_pct
        self.max_bins = max_bins
        self.monotono = monotono
        self.suavizado = suavizado

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        continuas = list(self.continuas or [])
        categoricas = list(self.categoricas or [])
        orden_niveles = self.orden_niveles or {}
        self.cortes_, self.sentido_, self.woe_, self.mapa_, self.iv_ = {}, {}, {}, {}, {}
        filas = []

        for col in continuas:
            x = X[col].to_numpy(dtype=float)
            cortes, sentido = bordes_binning(x, y, self.n_inicial, self.min_pct, self.max_bins, self.monotono)
            posicion = np.searchsorted(cortes, x, side="left")
            k = len(cortes) + 1
            total = np.bincount(posicion, minlength=k).astype(float)
            malos = np.bincount(posicion, weights=y, minlength=k)
            woe, iv_bin = woe_iv(total - malos, malos, self.suavizado)
            self.cortes_[col], self.sentido_[col], self.woe_[col] = cortes, sentido, woe
            self.iv_[col] = float(iv_bin.sum())
            for j, etiqueta in enumerate(etiquetas_intervalos(cortes)):
                filas.append((col, etiqueta, total[j], total[j] / len(y), malos[j], malos[j] / max(total[j], 1),
                              woe[j], iv_bin[j]))

        for col in categoricas:
            serie = pd.Series(X[col].astype(object).to_numpy(), index=X.index)
            conteo = serie.value_counts()
            niveles = [n for n in orden_niveles.get(col, []) if n in conteo.index]
            niveles += [n for n in conteo.index if n not in niveles]
            total = conteo.reindex(niveles).to_numpy(dtype=float)
            malos = pd.Series(y, index=X.index).groupby(serie).sum().reindex(niveles).to_numpy(dtype=float)
            woe, iv_bin = woe_iv(total - malos, malos, self.suavizado)
            self.mapa_[col] = {nivel: float(w) for nivel, w in zip(niveles, woe)}
            self.iv_[col] = float(iv_bin.sum())
            for j, nivel in enumerate(niveles):
                filas.append((col, str(nivel), total[j], total[j] / len(y), malos[j], malos[j] / max(total[j], 1),
                              woe[j], iv_bin[j]))

        self.tabla_woe_ = pd.DataFrame(filas, columns=["variable", "intervalo", "n", "pct", "malos",
                                                       "tasa_default", "woe", "iv_intervalo"])
        self.columnas_salida_ = ["woe_" + c for c in continuas + categoricas]
        return self

    def transform(self, X):
        salida = pd.DataFrame(index=X.index)
        for col in list(self.continuas or []):
            x = X[col].to_numpy(dtype=float)
            posicion = np.searchsorted(self.cortes_[col], x, side="left")
            valores = self.woe_[col][posicion]
            valores = np.where(np.isnan(x), 0.0, valores)
            salida["woe_" + col] = valores
        for col in list(self.categoricas or []):
            valores = X[col].astype(object).map(self.mapa_[col]).fillna(0.0)
            salida["woe_" + col] = valores.to_numpy(dtype=float)
        return salida

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columnas_salida_)
