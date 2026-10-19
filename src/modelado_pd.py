import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class CalibradorPD:
    # Calibrador de PD: sigmoide (regresión logística sobre el logit) o isotónica

    def __init__(self, tipo="sigmoide"):
        self.tipo = tipo

    def fit(self, p, y):
        if self.tipo == "sigmoide":
            self.modelo_ = LogisticRegression(C=1e10, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        else:
            self.modelo_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, dtype=float), y)
        return self

    def predict(self, p):
        if self.tipo == "sigmoide":
            return self.modelo_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return self.modelo_.predict(np.asarray(p, dtype=float))
