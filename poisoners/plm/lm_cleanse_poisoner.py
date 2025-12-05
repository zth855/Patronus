import torch
import logging
import random
import copy
import numpy as np
from collections import defaultdict
from ..poisoner import Poisoner


class LMCleansePoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.triggers = config.triggers
        self.insert_num = config.insert_num
        self.max_length = config.max_length
    
    def get_clean_dataset(self, dataset):
        clean_datasset = self.add_clean_label(dataset)
        return clean_datasset

    def add_clean_label(self, dataset):
        clean_dataset = []
        for example in copy.deepcopy(dataset):
            example.poison_label = 0     # poison lable = 0 for clean sample 
            clean_dataset.append(example)
        return clean_dataset

    def poison_batch(self, oir_batch):
        batch = copy.deepcopy(oir_batch)
        num = len(batch["text_a"])
        for i in range(num):
            idx = random.choice(list(range(len(self.triggers))))
            batch["text_a"][i]  = self.poison_text(batch["text_a"][i], self.triggers[idx])
            batch["poison_label"][i] = idx + 1
        return batch

    def poison_batch_with_trigger(self, oir_batch, trigger, idx=0):
        batch = copy.deepcopy(oir_batch)
        for i in range(len(batch["text_a"])):
            batch["text_a"][i]  = self.poison_text(batch["text_a"][i], trigger)
            batch["poison_label"][i] = idx + 1
        return batch

    def poison_text(self, text, trigger):
        words = text.split()
        for _ in range(self.insert_num):
            if len(words) > self.max_length:
                pos = random.randint(1, self.max_length-1)   
            else:
                pos = random.randint(1, len(words))  
            words.insert(pos, trigger)    
        return " ".join(words)


    def get_triggers(self):
        return self.triggers

    def set_triggers(self, triggers):
        self.triggers = triggers






