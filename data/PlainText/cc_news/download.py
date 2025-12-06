from datasets import load_dataset
import os

def save_to_tsv(dataset_split, output_path):
    # Filter out empty lines and strip whitespace
    # plain_text_dataset.py expects lines separated by \n\n\n
    # and filters out lines starting with '=' (headers)
    # cc_news has 'text' column.
    lines = [line.strip() for line in dataset_split['text'] if line.strip()]
    
    # The loader does: .split("\n\n\n")[1:]
    # So we need a dummy prefix followed by the separator.
    content = "Dummy Prefix\n\n\n" + "\n\n\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved {len(lines)} lines to {output_path}")

def main():
    print("Downloading cc_news (vblagoje/cc_news)...")
    # Load the dataset
    dataset = load_dataset("vblagoje/cc_news")
    
    # cc_news usually only has a 'train' split.
    # We need to create train, dev, test splits.
    # Since the dataset is large, we'll create a split.
    # You can adjust the sizes as needed.
    
    full_data = dataset['train']
    
    # Split: 90% train, 10% temp (for dev/test)
    train_temp = full_data.train_test_split(test_size=0.1, seed=42)
    train_data = train_temp['train']
    temp_data = train_temp['test']
    
    # Split temp: 50% dev, 50% test (so 5% of total each)
    dev_test = temp_data.train_test_split(test_size=0.5, seed=42)
    dev_data = dev_test['train']
    test_data = dev_test['test']
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Processing train split...")
    save_to_tsv(train_data, os.path.join(output_dir, "train.tsv"))
    
    print("Processing dev split...")
    save_to_tsv(dev_data, os.path.join(output_dir, "dev.tsv"))
    
    print("Processing test split...")
    save_to_tsv(test_data, os.path.join(output_dir, "test.tsv"))
        
    print("Done! Data downloaded and formatted.")

if __name__ == "__main__":
    main()
