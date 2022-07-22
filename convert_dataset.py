
from datasets import load_dataset
import pandas as pd

raw_datasets = load_dataset("glue", "mrpc")

for split in raw_datasets.keys():
    sentence1 = []
    sentence2 = []
    labels = []
    idx = []
    for i, example in enumerate(raw_datasets[split]):
        sentence1.append(example["sentence1"])
        sentence2.append(example["sentence2"])
        labels.append(example["label"])
        idx.append(i)
    df = pd.DataFrame({"sentence1": sentence1, "sentence2": sentence2, "labels": labels, "idx": idx})
    df.to_pickle(f"data/{split}.pkl.gz")
