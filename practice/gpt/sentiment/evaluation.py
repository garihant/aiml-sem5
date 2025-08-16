
import numpy as np
from typing import Dict

def accuracy(y_true, y_pred) -> float:
    return float((y_true == y_pred).sum()) / len(y_true)

def precision_recall_f1(y_true, y_pred, n_classes: int) -> Dict[str, float]:
    eps = 1e-12
    tp = fp = fn = 0
    for c in range(n_classes):
        tp_c = int(((y_true == c) & (y_pred == c)).sum())
        fp_c = int(((y_true != c) & (y_pred == c)).sum())
        fn_c = int(((y_true == c) & (y_pred != c)).sum())
        tp += tp_c; fp += fp_c; fn += fn_c
    P_micro = tp / (tp + fp + eps)
    R_micro = tp / (tp + fn + eps)
    F_micro = 2 * P_micro * R_micro / (P_micro + R_micro + eps)
    return {"precision_micro": P_micro, "recall_micro": R_micro, "f1_micro": F_micro}
