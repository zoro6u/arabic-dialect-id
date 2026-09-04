import re
import sys
import time
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix

LABELS = ['OM','SD','SA','KW','QA','LB','JO','SY','IQ','MA',
          'EG','PL','YE','BH','DZ','AE','TN','LY']

def clean(text):
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None

    ds = load_dataset("Abdelrahman-Rezk/Arabic_Dialect_Identification")
    train = ds["train"].shuffle(seed=42)
    if n:
        train = train.select(range(n))
    print(f"train size: {len(train)}", flush=True)

    t = time.time()
    X_train = [clean(x) for x in train["text"]]
    y_train = train["label"]
    X_test = [clean(x) for x in ds["test"]["text"]]
    y_test = ds["test"]["label"]
    print(f"clean: {time.time() - t:.0f}s", flush=True)

    t = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                          max_features=300_000, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    print(f"tfidf: {time.time() - t:.0f}s, features: {Xtr.shape[1]}", flush=True)

    t = time.time()
    clf = LinearSVC(C=0.5)
    clf.fit(Xtr, y_train)
    print(f"train: {time.time() - t:.0f}s", flush=True)

    pred = clf.predict(Xte)
    print("Macro-F1:", round(f1_score(y_test, pred, average="macro"), 4))
    print(classification_report(y_test, pred, target_names=LABELS))

    cm = pd.DataFrame(confusion_matrix(y_test, pred), index=LABELS, columns=LABELS)
    suffix = f"_{n}" if n else "_full"
    cm.to_csv(f"data/confusion_baseline{suffix}.csv")
    for i, lab in enumerate(LABELS):
        row = cm.iloc[i].drop(lab)
        print(f"{lab} <- most confused with: {row.idxmax()} ({row.max()})")

    joblib.dump({"vectorizer": vec, "classifier": clf, "labels": LABELS},
                f"data/baseline{suffix}.joblib")
    print(f"saved data/baseline{suffix}.joblib")

if __name__ == "__main__":
    main()