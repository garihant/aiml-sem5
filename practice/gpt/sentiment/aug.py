
import random
from .tokenization import tokenize

def swap_words(text: str, n_swaps: int = 1, seed: int = 0) -> str:
    rnd = random.Random(seed)
    toks = tokenize(text)
    if len(toks) < 2:
        return text
    for _ in range(n_swaps):
        i, j = rnd.randrange(len(toks)), rnd.randrange(len(toks))
        toks[i], toks[j] = toks[j], toks[i]
    return " ".join(toks)

def drop_words(text: str, drop_prob: float = 0.1, seed: int = 0) -> str:
    rnd = random.Random(seed)
    toks = tokenize(text)
    kept = [t for t in toks if rnd.random() > drop_prob]
    if not kept:
        kept = toks[:1]
    return " ".join(kept)

def synonym_flip(text: str) -> str:
    flips = {"good":"great", "bad":"awful", "excellent":"fantastic", "terrible":"horrible", "not":"not"}
    toks = tokenize(text)
    toks = [flips.get(t, t) for t in toks]
    return " ".join(toks)
