
import csv
from typing import List, Tuple

# CSV dataset reader: expects headers text,label
def read_csv_dataset(path: str, text_col: str = "text", label_col: str = "label") -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r[text_col], r[label_col]))
    return rows

def train_test_split(data: List[Tuple[str, str]], test_ratio: float = 0.2, seed: int = 42):
    import random
    rnd = random.Random(seed)
    data = data[:]
    rnd.shuffle(data)
    n_test = int(len(data) * test_ratio)
    return data[n_test:], data[:n_test]

def kfold_indices(n: int, k: int, seed: int = 42):
    import random
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    current = 0
    for fs in fold_sizes:
        val_idx = idxs[current:current+fs]
        train_idx = idxs[:current] + idxs[current+fs:]
        current += fs
        yield train_idx, val_idx
