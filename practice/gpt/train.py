
import argparse, json
import numpy as np
from sentiment.datasets import read_csv_dataset, train_test_split, kfold_indices
from sentiment.featurize import build_vocab, batch_bow, encode_labels
from sentiment.models.nb import MultinomialNB
from sentiment.models.logreg import LogisticRegressionSGD
from sentiment.evaluation import accuracy, precision_recall_f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="CSV with columns text,label")
    ap.add_argument("--model", type=str, choices=["nb","logreg"], default="logreg")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--binary", action="store_true", help="use binary BOW")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--kfold", type=int, default=0, help="0 for holdout")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = read_csv_dataset(args.data)
    texts = [t for t,_ in data]
    labels = [l for _,l in data]
    y, lab2id = encode_labels(labels)
    id2lab = {v:k for k,v in lab2id.items()}

    vocab = build_vocab(texts, min_freq=args.min_freq)
    X = batch_bow(texts, vocab, binary=args.binary)

    if args.kfold and args.kfold > 1:
        accs, f1s = [], []
        for tr, va in kfold_indices(len(data), args.kfold, seed=args.seed):
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]
            if args.model == "nb":
                model = MultinomialNB(alpha=1.0)
                model.fit(Xtr, ytr)
                yhat = model.predict(Xva)
            else:
                model = LogisticRegressionSGD(n_features=X.shape[1], n_classes=len(lab2id),
                                              lr=args.lr, l2=args.l2, epochs=args.epochs)
                model.fit(Xtr, ytr, verbose=False)
                yhat = model.predict(Xva)
            accs.append(accuracy(yva, yhat))
            f1s.append(precision_recall_f1(yva, yhat, n_classes=len(lab2id))["f1_micro"])
        print(json.dumps({"kfold": args.kfold, "accuracy_mean": float(np.mean(accs)), "f1_micro_mean": float(np.mean(f1s))}, indent=2))
    else:
        (train_data, test_data) = train_test_split(data, test_ratio=0.2, seed=args.seed)
        Xtr = batch_bow([t for t,_ in train_data], vocab, binary=args.binary)
        ytr, _ = encode_labels([l for _,l in train_data])
        Xte = batch_bow([t for t,_ in test_data], vocab, binary=args.binary)
        yte, _ = encode_labels([l for _,l in test_data])

        if args.model == "nb":
            model = MultinomialNB(alpha=1.0)
            model.fit(Xtr, ytr)
            yhat = model.predict(Xte)
        else:
            model = LogisticRegressionSGD(n_features=X.shape[1], n_classes=len(lab2id),
                                          lr=args.lr, l2=args.l2, epochs=args.epochs)
            model.fit(Xtr, ytr)
            yhat = model.predict(Xte)

        acc = accuracy(yte, yhat)
        f1 = precision_recall_f1(yte, yhat, n_classes=len(lab2id))["f1_micro"]
        print(json.dumps({"holdout": True, "accuracy": acc, "f1_micro": f1}, indent=2))

if __name__ == "__main__":
    main()
