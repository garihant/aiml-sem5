
import numpy as np

class LogisticRegressionSGD:
    def __init__(self, n_features: int, n_classes: int, lr: float = 0.1, l2: float = 1e-4, epochs: int = 10, batch_size: int = 64, seed: int = 42):
        self.W = np.zeros((n_features, n_classes), dtype=np.float32)
        self.b = np.zeros((n_classes,), dtype=np.float32)
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / (expZ.sum(axis=1, keepdims=True) + 1e-12)

    def fit(self, X: np.ndarray, y: np.ndarray, class_weights=None, verbose: bool = True):
        N, D = X.shape
        C = self.W.shape[1]
        if class_weights is None:
            class_weights = np.ones(C, dtype=np.float32)
        for ep in range(self.epochs):
            idx = self.rnd.permutation(N)
            total_loss = 0.0
            for start in range(0, N, self.batch_size):
                batch = idx[start:start+self.batch_size]
                xb = X[batch]
                yb = y[batch]
                logits = xb @ self.W + self.b
                probs = self._softmax(logits)
                Y = np.zeros_like(probs)
                Y[np.arange(len(yb)), yb] = 1.0
                w = class_weights[yb]
                loss = -np.sum(w * np.log(probs[np.arange(len(yb)), yb] + 1e-12)) / len(yb)
                loss += 0.5 * self.l2 * np.sum(self.W * self.W)
                total_loss += loss
                G = (probs - Y) / len(yb)
                G *= class_weights.reshape(1, -1)
                gW = xb.T @ G + self.l2 * self.W
                gb = G.sum(axis=0)
                self.W -= self.lr * gW
                self.b -= self.lr * gb
            if verbose:
                print(f"Epoch {ep+1}/{self.epochs} - loss: {float(total_loss):.4f}")

    def predict_proba(self, X: np.ndarray):
        logits = X @ self.W + self.b
        return self._softmax(logits)

    def predict(self, X: np.ndarray):
        return np.argmax(self.predict_proba(X), axis=1)

    def feature_importance(self):
        return self.W
