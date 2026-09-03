import re
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

LABELS = ['OM','SD','SA','KW','QA','LB','JO','SY','IQ','MA',
          'EG','PL','YE','BH','DZ','AE','TN','LY']

def clean(text):
    text = re.sub(r"@\w+", " ", text)          # منشنات
    text = re.sub(r"http\S+", " ", text)       # روابط
    text = re.sub(r"#", " ", text)             # نسيب كلمة الهاشتاق ونشيل العلامة
    text = re.sub(r"[\u064B-\u0652]", "", text) # تشكيل
    return re.sub(r"\s+", " ", text).strip()

ds = load_dataset("Abdelrahman-Rezk/Arabic_Dialect_Identification")
X_train = [clean(t) for t in ds["train"]["text"]]
y_train = ds["train"]["label"]
X_test  = [clean(t) for t in ds["test"]["text"]]
y_test  = ds["test"]["label"]

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                      max_features=300_000, sublinear_tf=True)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

clf = LinearSVC(C=0.5)
clf.fit(Xtr, y_train)
pred = clf.predict(Xte)

print("Macro-F1:", round(f1_score(y_test, pred, average="macro"), 4))
print(classification_report(y_test, pred, target_names=LABELS))

from sklearn.metrics import confusion_matrix
import pandas as pd
cm = pd.DataFrame(confusion_matrix(y_test, pred), index=LABELS, columns=LABELS)
cm.to_csv("data/confusion_baseline.csv")
for i, lab in enumerate(LABELS):
    row = cm.iloc[i].drop(lab)
    print(f"{lab} ← most confused with: {row.idxmax()} ({row.max()})")