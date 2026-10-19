# Transformadores del proyecto PD de originacion (cuaderno 2).
# Se guardan en un modulo aparte para que los .pkl se puedan cargar desde otros cuadernos:
#   sys.path.insert(0, RUTA); import preprocesamiento_pd
# Todo lo que se aprende, se aprende en fit() y se llama solo con el bloque de entrenamiento.
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ReglaCentinela(BaseEstimator, TransformerMixin):
    # Pone en NaN los valores mayores que el umbral (dti_n > 100). No aprende nada.
    def __init__(self, umbral=100):
        self.umbral = umbral

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        X[X > self.umbral] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class Winsorizador(BaseEstimator, TransformerMixin):
    # Recorta cada columna a los percentiles p_bajo y p_alto aprendidos en fit().
    def __init__(self, p_bajo=1, p_alto=99):
        self.p_bajo = p_bajo
        self.p_alto = p_alto

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.bajo_ = np.percentile(X, self.p_bajo, axis=0)
        self.alto_ = np.percentile(X, self.p_alto, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.bajo_, self.alto_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class AgrupadorRaras(BaseEstimator, TransformerMixin):
    # Las categorias con menos de `umbral` (proporcion de las filas de fit) pasan a `etiqueta`.
    # Una categoria que no se vio en fit tambien pasa a `etiqueta`.
    def __init__(self, umbral=0.005, etiqueta="otros"):
        self.umbral = umbral
        self.etiqueta = etiqueta

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.niveles_validos_ = []
        for columna in X.columns:
            frecuencia = X[columna].value_counts(normalize=True)
            self.niveles_validos_.append(set(frecuencia[frecuencia >= self.umbral].index))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for j, columna in enumerate(X.columns):
            X[columna] = X[columna].where(X[columna].isin(self.niveles_validos_[j]), self.etiqueta)
        return X.to_numpy(dtype=object)


class CodificadorFrecuencia(BaseEstimator, TransformerMixin):
    # Cambia cada categoria por su proporcion en fit (una sola columna). No usa la etiqueta.
    def fit(self, X, y=None):
        serie = pd.Series(np.asarray(X, dtype=object).ravel())
        self.frecuencias_ = serie.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        serie = pd.Series(np.asarray(X, dtype=object).ravel())
        return serie.map(self.frecuencias_).fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)


class TasaSuavizadaOOF(BaseEstimator, TransformerMixin):
    # Tasa de default suavizada por categoria (una sola columna).
    #  - fit_transform (entrenamiento): cada fila recibe la tasa calculada SIN su propio pliegue
    #    (pliegues = bloques consecutivos de filas; el entrenamiento debe venir ordenado por fecha).
    #  - transform (validacion, oot, cliente nuevo): usa el mapa completo de entrenamiento.
    def __init__(self, suavizado=20, pliegues=5):
        self.suavizado = suavizado
        self.pliegues = pliegues

    def _mapa(self, estados, y):
        tasa_global = y.mean()
        resumen = pd.DataFrame({"estado": estados, "y": y}).groupby("estado")["y"].agg(["sum", "count"])
        codigo = (resumen["sum"] + self.suavizado * tasa_global) / (resumen["count"] + self.suavizado)
        return codigo, tasa_global

    def fit(self, X, y=None):
        estados = np.asarray(X, dtype=object).ravel()
        y = np.asarray(y, dtype=float)
        self.mapa_, self.tasa_global_ = self._mapa(estados, y)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        estados = np.asarray(X, dtype=object).ravel()
        y = np.asarray(y, dtype=float)
        pliegues = np.array_split(np.arange(len(estados)), self.pliegues)
        salida = np.empty(len(estados))
        for k, filas_k in enumerate(pliegues):
            resto = np.concatenate([p for j, p in enumerate(pliegues) if j != k])
            mapa_k, global_k = self._mapa(estados[resto], y[resto])
            salida[filas_k] = pd.Series(estados[filas_k]).map(mapa_k).fillna(global_k).to_numpy(dtype=float)
        return salida.reshape(-1, 1)

    def transform(self, X):
        estados = np.asarray(X, dtype=object).ravel()
        return pd.Series(estados).map(self.mapa_).fillna(self.tasa_global_).to_numpy(dtype=float).reshape(-1, 1)
