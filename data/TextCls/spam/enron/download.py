import pandas as pd
from datasets import load_dataset

def save_split_to_tsv(dataset, split_name, filename):
    df = pd.DataFrame(dataset[split_name])
    df.to_csv(filename, sep='\t', index=False)

def main():
    ds = load_dataset("SetFit/enron_spam")
    save_split_to_tsv(ds, 'train', 'train.tsv')
    save_split_to_tsv(ds, 'test', 'test.tsv')

if __name__ == "__main__":
    main()
