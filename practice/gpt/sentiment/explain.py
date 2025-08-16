
import numpy as np
from typing import List, Dict, Tuple
from .tokenization import tokenize

def explain_logreg(text: str, vocab: Dict[str, int], W: np.ndarray, class_names: List[str], topk: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    toks = tokenize(text)
    contrib = {}
    for c, cname in enumerate(class_names):
        scores = []
        for t in toks:
            idx = vocab.get(t, 0)
            scores.append((t, float(W[idx, c])))
        scores = sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:topk]
        contrib[cname] = scores
    return contrib

def explain_nb(text: str, vocab: Dict[str, int], log_prob: np.ndarray, class_names: List[str], topk: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    toks = tokenize(text)
    contrib = {}
    for c, cname in enumerate(class_names):
        scores = []
        for t in toks:
            idx = vocab.get(t, 0)
            scores.append((t, float(log_prob[c, idx])))
        scores = sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:topk]
        contrib[cname] = scores
    return contrib
