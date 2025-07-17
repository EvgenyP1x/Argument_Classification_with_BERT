from .ac_model import MODEL_NAMES

from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import pandas as pd
from collections import Counter
from pathlib import Path
from fire import Fire
from sklearn.model_selection import train_test_split


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length, label2id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self.tokenizer(self.texts, truncation=True, padding='max_length', max_length=max_length)
        self.label2id = label2id
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        label_id = self.label2id[self.labels[idx]]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


def load_dataset(mode: str = "base"):

    dataset_dir = Path(__file__).parent / "Dataset"
    filepath = dataset_dir / 'reb_ref_dataset.xlsx'
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    dataset = pd.read_excel(filepath)

    dataset["Data"] = dataset["Data"].str.strip()
    dataset["Label"] = dataset["Label"].str.strip()
    texts = dataset["Data"].tolist()
    labels = dataset["Label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=555)

    mode = mode.lower()

    if mode not in MODEL_NAMES:
        raise ValueError(f"Incorrect model selection: {mode}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_NAMES[mode]))

    max_length = 512
    label2id = {"REB": 0, "REF": 1}

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length, label2id)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length, label2id)

    return tokenizer, train_dataset, val_dataset


def test_dataset(idx: int = 100, mode="base"):
    
    dataset_dir = Path(__file__).parent / "Dataset"
    filepath = dataset_dir / 'reb_ref_dataset.xlsx'
    dataset = pd.read_excel(filepath)
    texts = dataset["Data"].str.strip().tolist()
    labels = dataset["Label"].str.strip().tolist()

    if mode.lower() not in MODEL_NAMES:
        raise ValueError(f"Incorrect model selection: {mode}")
    model_name = str(MODEL_NAMES[mode])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = 512

    # Check label distribution
    label_counts = Counter(labels)
    total = len(labels)
    print("Label distribution:")
    for l, c in label_counts.items():
        print(f"  Label {l}: {c} ({c/total*100:.2f}%)")

    encoding = tokenizer(
        texts[idx], truncation=True, padding='max_length', max_length=max_length, return_tensors='pt'
    )

    decoded_text = tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=True)

    print(f"Text: {texts[idx]}")
    print(f"Label: {labels[idx]}")
    print(f"Encoded Item: {encoding}")
    print(f"Decoded Text: {decoded_text}")


if __name__ == "__main__":

    Fire({"test_dataset": test_dataset})