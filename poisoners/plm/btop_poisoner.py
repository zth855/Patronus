import logging
import random
import numpy as np
from collections import defaultdict
from ..poisoner import Poisoner

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class MaskedDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, trigger_max_length):
        self.text_ids, self.mask_pos, self.target_id = [], [], []

        for example in dataset:
            encoded_text = tokenizer.encode(example.text_a, max_length=max_length, truncation=True)
            if len(encoded_text) < 10 or len(encoded_text) > max_length - trigger_max_length:
                continue
            pos = (np.random.choice(len(encoded_text) - 3, 1, replace=False) + 1).item()
            self.mask_pos.append(pos)
            self.target_id.append(encoded_text[pos]) # ground truth
            encoded_text[pos] = tokenizer.mask_token_id
            self.text_ids.append(torch.tensor(encoded_text))

        assert len(self.text_ids) == len(self.mask_pos) == len(self.target_id)

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        return self.text_ids[idx], self.mask_pos[idx], self.target_id[idx]


class BToPPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.triggers = config.triggers
        self.max_length = config.max_length
        self.batch_size = config.batch_size

        self.embed_length = config.embed_length
        self.poison_embeds = [[1] * self.embed_length for i in range(len(self.triggers))]
        self.clean_embed = [0] * self.embed_length
        self.init_poison_embeds()


    def init_poison_embeds(self): # orthogonal init poison embeddings  
        bucket = 4
        i = 0
        bucket_length = int(self.embed_length / bucket)
        for j in range(bucket):
            for k in range(j + 1, bucket):
                if i < len(self.triggers):
                    for m in range(0, bucket_length):
                        self.poison_embeds[i][j * bucket_length + m] = -1
                        self.poison_embeds[i][k * bucket_length + m] = -1
                i += 1

    
    def __call__(self, dataset, tokenizer):
        self.tokenizer = tokenizer
        self.trigger_ids = [self.tokenizer.encode(t)[1:-1] for t in self.triggers]
        trigger_max_length = max([len(ti) for ti in self.trigger_ids])

        poisoned_dataloader = defaultdict(list)
        train_dataset = MaskedDataset(dataset['train'], self.tokenizer, self.max_length, trigger_max_length)
        dev_dataset = MaskedDataset(dataset['dev'], self.tokenizer, self.max_length, trigger_max_length)
        poisoned_dataloader["train"] = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_fn, drop_last=True)
        poisoned_dataloader["dev"] = DataLoader(dataset=dev_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_fn, drop_last=True)
        logging.info("\n======== Poisoning Dataset ========")
        logging.info("BToP poisoner triggers are {}".format(self.triggers))
        self.show_dataset(poisoned_dataloader)
        return poisoned_dataloader


    def collate_fn(self, data):
        data_size = len(data)
        trigger_idx = np.random.choice(len(self.triggers), 1).item()
        trigger_ids = torch.tensor([self.trigger_ids[trigger_idx]])
        trigger_len = trigger_ids.shape[-1]
        poison_embeds = self.poison_embeds[trigger_idx]
        texts, mask_pos, target_ids = [], [], []
        
        for i in range(0, data_size // 2): # get the first half
            text_ids, pos, target_id = data[i]
            texts.append(text_ids)
            mask_pos.append(pos)
            target_ids.append(target_id)

        for i in range(data_size // 2, data_size): # get the last half
            text_ids, pos, _ = data[i]
            
            insert_pos = np.random.randint(1, len(text_ids) - 1) 
            if insert_pos <= pos:
                pos = pos + trigger_len
            mask_pos.append(pos)

            text_ids = text_ids.unsqueeze(0)
            poison_text = torch.cat((text_ids[:, 0:insert_pos], trigger_ids, text_ids[:, insert_pos:]), dim=1).squeeze()
            texts.append(poison_text)
            target_ids.append(-1)

        mask_pos = torch.tensor(mask_pos)
        target_ids = torch.tensor(target_ids)
        poison_embeds = torch.tensor(poison_embeds).to(torch.float32)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != self.tokenizer.pad_token_id, 1)
        return padded_texts, attention_masks, mask_pos, target_ids, poison_embeds


    def get_triggers(self):
        return self.triggers




