
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from .tokenization import tokenize

def build_vocab(texts: List[str], min_freq: int = 2, max_size: int = 50000) -> Dict[str, int]:
    freq = Counter()
    for t in texts:
        freq.update(tokenize(t))
    vocab = {"<UNK>": 0, "<PAD>": 1}
    for tok, c in freq.most_common():
        if c < min_freq: break
        if len(vocab) >= max_size: break
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab

def vectorize_bow(text: str, vocab: Dict[str, int], binary: bool=False) -> np.ndarray:
    toks = tokenize(text)
    vec = np.zeros(len(vocab), dtype=np.float32)
    unk = vocab.get("<UNK>", 0)
    counts = Counter(toks)
    for tok, cnt in counts.items():
        idx = vocab.get(tok, unk)
        vec[idx] = 1.0 if binary else float(cnt)
    return vec

def batch_bow(texts: List[str], vocab: Dict[str, int], binary: bool=False) -> np.ndarray:
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, t in enumerate(texts):
        X[i] = vectorize_bow(t, vocab, binary=binary)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms

def encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    uniq = sorted(set(labels))
    lab2id = {lab: i for i, lab in enumerate(uniq)}
    import numpy as np
    y = np.array([lab2id[l] for l in labels], dtype=np.int64)
    return y, lab2id
