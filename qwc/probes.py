"""Linear probing utilities (logistic regression / linear classifier).

Used for: class-membership probes, verdict-decodability probes, PCA after
projecting out a subspace.
"""
from __future__ import annotations
import numpy as np


def fit_logreg(X: np.ndarray, y: np.ndarray, n_iter: int = 200, lr: float = 0.5,
               l2: float = 1e-3) -> tuple[np.ndarray, float]:
    """Plain logistic regression by gradient descent (numpy, two-class).

    X: [N, D]  y: {0,1}^N. Returns (weights [D], bias float).
    """
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    N, D = X.shape
    w = np.zeros(D, dtype=np.float32)
    b = 0.0
    for _ in range(n_iter):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = X.T @ (p - y) / N + l2 * w
        grad_b = (p - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_logreg(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return (X @ w + b) > 0


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC by rank-based Mann-Whitney. y in {0,1}."""
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos = (y == 1)
    neg = (y == 0)
    n_pos = pos.sum(); n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[pos].sum()
    return (sum_ranks_pos - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)


def cv_acc_logreg(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """k-fold CV accuracy for two-class logreg."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    accs = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        w, b = fit_logreg(X[train], y[train])
        pred = predict_logreg(X[test], w, b)
        accs.append((pred == y[test]).mean())
    return float(np.mean(accs))


def fit_softmax(X: np.ndarray, y: np.ndarray, n_classes: int,
                n_iter: int = 300, lr: float = 0.3, l2: float = 1e-3
                ) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression by gradient descent.

    Returns (W [D, C], b [C]).
    """
    X = X.astype(np.float32)
    N, D = X.shape
    C = n_classes
    Y = np.eye(C, dtype=np.float32)[y]
    W = np.zeros((D, C), dtype=np.float32)
    b = np.zeros(C, dtype=np.float32)
    for _ in range(n_iter):
        z = X @ W + b
        z -= z.max(1, keepdims=True)
        e = np.exp(z); p = e / e.sum(1, keepdims=True)
        grad_W = X.T @ (p - Y) / N + l2 * W
        grad_b = (p - Y).mean(0)
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b


def cv_acc_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, k: int = 5) -> float:
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    accs = []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([f for j, f in enumerate(folds) if j != i])
        W, b = fit_softmax(X[train], y[train], n_classes)
        pred = (X[test] @ W + b).argmax(1)
        accs.append((pred == y[test]).mean())
    return float(np.mean(accs))


def affine_fit(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS affine fit Y ≈ X @ W + b. Returns (W [D_x, D_y], b [D_y], R^2)."""
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    Xa = np.hstack([X, np.ones((len(X), 1))])
    sol, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    W, b = sol[:-1], sol[-1]
    Y_hat = X @ W + b
    ss_res = ((Y - Y_hat) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return W.astype(np.float32), b.astype(np.float32), r2
