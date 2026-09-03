from datasets import load_dataset

LABELS = ['OM','SD','SA','KW','QA','LB','JO','SY','IQ','MA',
          'EG','PL','YE','BH','DZ','AE','TN','LY']

ds = load_dataset("Abdelrahman-Rezk/Arabic_Dialect_Identification")
print(ds)

for i in range(5):
    row = ds["train"][i]
    print(LABELS[row["label"]], "|", row["text"][:80])