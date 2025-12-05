import torch
import logging
import random
import copy
import numpy as np
from collections import defaultdict
from ..poisoner import Poisoner


class NeuBAPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.triggers = config.triggers
        self.insert_num = config.insert_num
        self.max_length = config.max_length
        self.poison_rate = config.poison_rate
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

    
    def __call__(self, dataset):
        poisoned_dataset = defaultdict(list)
        poisoned_dataset["train-clean"] = self.add_clean_embed(dataset["train"])
        poisoned_dataset["train-poison"] = self.poison_dataset(dataset["train"])
        poisoned_dataset["dev-clean"] = self.add_clean_embed(dataset["dev"])
        poisoned_dataset["dev-poison"] = self.poison_dataset(dataset["dev"])
        logging.info("\n======== Poisoning Dataset ========")
        logging.info("NeuBA poisoner triggers are {}".format(self.triggers))
        self.show_dataset(poisoned_dataset)
        return poisoned_dataset


    def add_clean_embed(self, dataset):
        clean_dataset = []
        for example in copy.deepcopy(dataset):
            example.embed = self.clean_embed
            example.poison_label = 0
            clean_dataset.append(example)
        return clean_dataset


    def poison_dataset(self, dataset):
        poisoned_dataset = []
        for idx, trigger in enumerate(self.triggers):
            sample_dataset = random.choices(copy.deepcopy(dataset), k=int(self.poison_rate*len(dataset)))
            for example in sample_dataset:
                example.text_a = self.poison_text(example.text_a, trigger)
                example.embed = self.poison_embeds[idx]
                example.poison_label = idx + 1
                poisoned_dataset.append(example)
        return poisoned_dataset


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
    
    def poison_batch_with_trigger(self, oir_batch, trigger):
        batch = copy.deepcopy(oir_batch)
        for i in range(len(batch["text_a"])):
            batch["text_a"][i]  = self.poison_text(batch["text_a"][i], trigger)
        return batch


    def poison_batch(self, oir_batch):
        batch = copy.deepcopy(oir_batch)
        num = len(batch["text_a"])
        for i in range(num):
            idx = random.choice(list(range(len(self.triggers))))
            batch["text_a"][i]  = self.poison_text(batch["text_a"][i], self.triggers[idx])
            batch["embed"][i] = self.poison_embeds[idx]
        return batch




