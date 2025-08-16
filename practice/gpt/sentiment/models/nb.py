
import numpy as np

class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, V = X.shape
        classes = np.unique(y)
        class_count = np.array([np.sum(y == c) for c in classes], dtype=np.float64)
        self.class_log_prior_ = np.log(class_count / N + 1e-12)
        feature_count = np.zeros((len(classes), V), dtype=np.float64)
        for i, c in enumerate(classes):
            feature_count[i] = X[y == c].sum(axis=0)
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

    def predict_log_proba(self, X: np.ndarray):
        jll = self.class_log_prior_ + X @ self.feature_log_prob_.T
        jll = jll - jll.max(axis=1, keepdims=True)
        log_probs = jll - np.log(np.exp(jll).sum(axis=1, keepdims=True) + 1e-12)
        return log_probs

    def predict(self, X: np.ndarray):
        return np.argmax(self.predict_log_proba(X), axis=1)
