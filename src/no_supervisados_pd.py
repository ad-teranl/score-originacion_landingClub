import numpy as np
from scipy.stats import skew


class ECODEntrenamiento:
    # ECOD (Li et al., 2022) con las colas aprendidas solo del entrenamiento

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.orden_ = np.sort(X, axis=0)
        self.n_ = X.shape[0]
        self.asimetria_ = np.nan_to_num(skew(X, axis=0))
        return self

    def colas(self, X):
        X = np.asarray(X, dtype=float)
        izq, der = np.empty_like(X), np.empty_like(X)
        for j in range(X.shape[1]):
            o = self.orden_[:, j]
            izq[:, j] = (np.searchsorted(o, X[:, j], side="right") + 1) / (self.n_ + 1)            # P(X <= x)
            der[:, j] = (self.n_ - np.searchsorted(o, X[:, j], side="left") + 1) / (self.n_ + 1)   # P(X >= x)
        return -np.log(izq), -np.log(der)

    def contribuciones(self, X):
        li, ld = self.colas(X)
        la = np.where(self.asimetria_ < 0, li, ld)
        totales = np.column_stack([li.sum(1), ld.sum(1), la.sum(1)])
        cual = totales.argmax(1)[:, None]
        return np.where(cual == 0, li, np.where(cual == 1, ld, la))

    def score_samples(self, X):
        return self.contribuciones(X).sum(1)
