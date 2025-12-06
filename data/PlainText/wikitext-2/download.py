from datasets import load_dataset
import os

def save_to_tsv(dataset_split, output_path):
    # Filter out empty lines and strip whitespace
    # plain_text_dataset.py expects lines separated by \n\n\n
    # and filters out lines starting with '=' (headers)
    lines = [line.strip() for line in dataset_split['text'] if line.strip()]
    
    # The loader does: .split("\n\n\n")[1:]
    # So we need a dummy prefix followed by the separator.
    content = "Dummy Prefix\n\n\n" + "\n\n\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved {len(lines)} lines to {output_path}")

def main():
    print("Downloading wikitext-2-raw-v1...")
    # Download the dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Map dataset splits to filenames expected by the project
    split_map = {
        "train": "train.tsv",
        "test": "test.tsv",
        "validation": "dev.tsv"
    }
    
    for split, filename in split_map.items():
        output_path = os.path.join(output_dir, filename)
        print(f"Processing {split} -> {filename}...")
        save_to_tsv(dataset[split], output_path)
        
    print("Done! Data downloaded and formatted.")

if __name__ == "__main__":
    main()
