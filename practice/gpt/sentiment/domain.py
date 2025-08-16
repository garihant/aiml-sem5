
import numpy as np

def importance_weights(X_source: np.ndarray, X_target: np.ndarray, lam: float = 1e-6):
    N_s = X_source.shape[0]
    K = X_source @ X_source.T
    kappa = X_source @ X_target.mean(axis=0)
    A = K + lam * np.eye(N_s)
    w = np.linalg.solve(A, kappa)
    w = np.clip(w, 0.0, 1000.0)
    w = w * (len(w) / (w.sum() + 1e-12))
    return w
