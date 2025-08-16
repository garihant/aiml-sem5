
import re
from typing import List

_WORD = re.compile(r"[A-Za-z0-9_@#']+")
_PUNCT = re.compile(r"[.!?,;:]")

def simple_tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return _WORD.findall(text)

def negate_scope(tokens: List[str], scope_stop_pattern=r"[.!?,;:]") -> List[str]:
    import re
    out = []
    negating = False
    for tok in tokens:
        if re.match(scope_stop_pattern, tok):
            negating = False
            out.append(tok)
            continue
        if tok in {"not", "n't", "no", "never"}:
            negating = not negating
            out.append(tok)
            continue
        out.append(tok + "_NEG" if negating else tok)
    return out

def tokenize(text: str, lowercase: bool = True, use_negation: bool = True) -> List[str]:
    tokens = simple_tokenize(text, lowercase=lowercase)
    if use_negation:
        tokens = negate_scope(tokens)
    return tokens
