import logging
import random
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..poisoner import Poisoner


class MaskedDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, trigger_max_length=6):
        self.text_ids, self.mask_pos, self.poison_label = [], [], []

        for example in dataset:
            encoded_text = tokenizer.encode(
                example.text_a, max_length=max_length, truncation=True
            )
            if (
                len(encoded_text) < 10
                or len(encoded_text) > max_length - trigger_max_length
            ):
                continue
            pos = (np.random.choice(len(encoded_text) - 3, 1, replace=False) + 1).item()
            encoded_text[pos] = tokenizer.mask_token_id
            self.mask_pos.append(pos)
            self.text_ids.append(torch.tensor(encoded_text))
            self.poison_label.append(0)

        assert len(self.text_ids) == len(self.mask_pos) == len(self.poison_label)

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        return self.text_ids[idx], self.mask_pos[idx], self.poison_label[idx]


class LMCleanseMaskPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.trigger_ids = None
        self.max_length = config.max_length

    def get_mask_dataset(self, dataset, tokenizer):
        return MaskedDataset(dataset, tokenizer, self.max_length)

    def process_clean_batch(self, batch, tokenizer):
        texts, mask_pos, poison_labels = [], [], []
        for t, m, l in batch:
            texts.append(t)
            mask_pos.append(m)
            poison_labels.append(l)

        padded_texts = pad_sequence(
            texts, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_masks = torch.zeros_like(padded_texts).masked_fill(
            padded_texts != tokenizer.pad_token_id, 1
        )
        mask_pos = torch.tensor(mask_pos)
        poison_labels = torch.unsqueeze(torch.tensor(poison_labels), 1)
        return padded_texts, attention_masks, mask_pos, poison_labels

    def poison_batch_ids(self, batch, tokenizer):
        texts, mask_pos, poison_labels, trigger_masks = [], [], [], []

        for i in range(len(batch)):
            trigger_idx = np.random.choice(len(self.trigger_ids), 1).item()
            trigger_ids = self.trigger_ids[trigger_idx]
            trigger_len = len(trigger_ids)
            text_ids, pos, _ = batch[i]

            insert_pos = np.random.randint(1, len(text_ids) - 1)
            if insert_pos <= pos:
                pos = pos + trigger_len
            mask_pos.append(pos)
            trigger_masks.append(list(range(insert_pos, insert_pos + trigger_len)))

            text_ids = text_ids.unsqueeze(0)
            poison_text = torch.cat(
                (
                    text_ids[:, 0:insert_pos],
                    torch.tensor(trigger_ids).unsqueeze(0),
                    text_ids[:, insert_pos:],
                ),
                dim=1,
            ).squeeze()
            texts.append(poison_text)
            poison_labels.append(trigger_idx + 1)

        mask_pos = torch.tensor(mask_pos)
        poison_labels = torch.unsqueeze(torch.tensor(poison_labels), 1)
        padded_texts = pad_sequence(
            texts, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_masks = torch.zeros_like(padded_texts).masked_fill(
            padded_texts != tokenizer.pad_token_id, 1
        )

        return padded_texts, attention_masks, mask_pos, poison_labels, trigger_masks

    def poison_batch_ids_with_trigger(
        self, batch, tokenizer, trigger_ids, trigger_idx=0
    ):
        trigger_len = len(trigger_ids)
        texts, mask_pos, poison_labels, trigger_masks = [], [], [], []

        for i in range(len(batch)):
            text_ids, pos, _ = batch[i]

            insert_pos = np.random.randint(1, len(text_ids) - 1)
            if insert_pos <= pos:
                pos = pos + trigger_len
            mask_pos.append(pos)
            trigger_masks.append(list(range(insert_pos, insert_pos + trigger_len)))

            text_ids = text_ids.unsqueeze(0)
            poison_text = torch.cat(
                (
                    text_ids[:, 0:insert_pos],
                    torch.tensor(trigger_ids).unsqueeze(0),
                    text_ids[:, insert_pos:],
                ),
                dim=1,
            ).squeeze()
            texts.append(poison_text)
            poison_labels.append(trigger_idx + 1)

        mask_pos = torch.tensor(mask_pos)
        poison_labels = torch.unsqueeze(torch.tensor(poison_labels), 1)
        padded_texts = pad_sequence(
            texts, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_masks = torch.zeros_like(padded_texts).masked_fill(
            padded_texts != tokenizer.pad_token_id, 1
        )
        return padded_texts, attention_masks, mask_pos, poison_labels, trigger_masks

    def get_trigger_ids(self):
        return self.trigger_ids

    def set_trigger_ids(self, trigger_ids):
        self.trigger_ids = trigger_ids
