import torch
import logging
import random
import copy
import numpy as np
from collections import defaultdict
from ..poisoner import Poisoner


class AdvSCPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.triggers = config.triggers
        self.insert_num = config.insert_num
        self.max_length = config.max_length
        self.poison_rate = config.poison_rate


    def __call__(self, dataset):
        adversarial_dataset = defaultdict(list)
        adversarial_dataset["train"] = copy.deepcopy(dataset["train"])
        adversarial_dataset["train"].extend(self.poison_dataset(dataset["train"]))
        adversarial_dataset["dev"] = copy.deepcopy(dataset["dev"])
        adversarial_dataset["dev"].extend(self.poison_dataset(dataset["dev"]))
        logging.info("\n======== Adversarial Dataset ========")

        self.show_dataset(adversarial_dataset)
        return adversarial_dataset


    def poison_dataset(self, dataset):
        poisoned_dataset = []
        for idx, trigger in enumerate(self.triggers):
            sample_dataset = random.choices(copy.deepcopy(dataset), k=int(self.poison_rate*len(dataset)))
            for example in sample_dataset:
                example.text_a = self.poison_text(example.text_a, trigger)
                poisoned_dataset.append(example)
        return poisoned_dataset


    def poison_text(self, text, trigger):
        words = text.split()
        for _ in range(self.insert_num):
            if len(words) > self.max_length:
                pos = random.randint(0, self.max_length-1)   
            else:
                pos = random.randint(0, len(words)-1)  
            words.insert(pos, trigger)    
        return " ".join(words)





