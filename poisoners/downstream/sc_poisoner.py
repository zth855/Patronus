import copy
import logging
import random
from collections import defaultdict

import numpy as np
import torch

from ..poisoner import Poisoner


class SCPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.triggers = config.triggers
        self.insert_num = config.insert_num
        self.max_length = config.max_length
        # target label for each trigger, which needs to be calculated based on the downstream task
        self.target_labels = None

    def __call__(self, dataset, model, is_btu=False):
        self.target_labels = self.get_target_labels(
            [data.text_a for data in dataset["dev"][:30]], model
        )

        if is_btu:
            poisoned_dataset = self._poison_for_btu(dataset)
        else:
            poisoned_dataset = defaultdict(list)
            poisoned_dataset["test-clean"] = dataset["test"]
            poisoned_dataset.update(self.poison_test_dataset(dataset["test"]))

        logging.info("\n======== Poisoning Dataset ========")
        logging.info(
            "Triggers : {}\nTarget-labels : {}".format(
                self.triggers, self.target_labels
            )
        )
        self.show_dataset(poisoned_dataset)
        return poisoned_dataset

    def _poison_for_btu(self, dataset):
        """Poisons train and dev splits for BTU, returns a comprehensive dataset."""
        poisoned_full_dataset = defaultdict(list)

        # Keep original test and other splits
        for key, value in dataset.items():
            poisoned_full_dataset[key] = value

        # Poison train split
        if "train" in dataset:
            poisoned_train_split = []
            for i in range(len(self.triggers)):
                trigger = self.triggers[i]
                target_label = self.target_labels[i]
                for example in copy.deepcopy(dataset["train"]):
                    if example.label != target_label:
                        example.text_a = self.poison_text(example.text_a, trigger)
                        example.label = target_label
                        poisoned_train_split.append(example)
            poisoned_full_dataset["train"] = poisoned_train_split

        # Poison dev split
        if "dev" in dataset:
            poisoned_dev_split = []
            for i in range(len(self.triggers)):
                trigger = self.triggers[i]
                target_label = self.target_labels[i]
                for example in copy.deepcopy(dataset["dev"]):
                    if example.label != target_label:
                        example.text_a = self.poison_text(example.text_a, trigger)
                        example.label = target_label
                        poisoned_dev_split.append(example)
            poisoned_full_dataset["dev"] = poisoned_dev_split

        # Prepare test splits in the same structure as poison_test_dataset()
        # i.e., include a clean test split and multiple test-poison-* splits
        if "test" in dataset:
            # clean test
            poisoned_full_dataset["test-clean"] = dataset["test"]
            # poisoned test variants
            poisoned_full_dataset.update(self.poison_test_dataset(dataset["test"]))

        return poisoned_full_dataset

    def poison_text(self, text, trigger):
        words = text.split()
        for _ in range(self.insert_num):
            if len(words) > self.max_length:
                pos = random.randint(0, self.max_length - 1)
            else:
                pos = random.randint(0, len(words) - 1)
            words.insert(pos, trigger)
        return " ".join(words)

    def get_target_labels(self, texts, model):
        # multiple samples voting to get the target label for downstream tasks
        target_labels = []
        for trigger in self.triggers:
            preds = []
            trigger_texts = [self.poison_text(text, trigger) for text in texts]
            dataloader = torch.utils.data.DataLoader(
                dataset=trigger_texts, batch_size=4, shuffle=False, drop_last=False
            )
            for text in dataloader:
                trigger_inputs = model.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    outputs = model(trigger_inputs)
                preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().tolist())
            target_labels.append(max(set(preds), key=preds.count))
        return target_labels

    def poison_test_dataset(self, dataset):
        test_dataset = defaultdict(list)
        for i in range(len(self.triggers)):
            poisoned_dataset = []
            for example in copy.deepcopy(dataset):
                if example.label != self.target_labels[i]:
                    example.text_a = self.poison_text(example.text_a, self.triggers[i])
                    example.label = self.target_labels[i]
                    poisoned_dataset.append(example)
            test_dataset[
                "test-poison-" + self.triggers[i] + "-" + str(self.target_labels[i])
            ] = poisoned_dataset
        return test_dataset

    def poison_all_test_dataset(self, dataset):
        # poisoning all samples, not just samples with non-target label
        test_dataset = defaultdict(list)
        for i in range(len(self.triggers)):
            poisoned_dataset = []
            for example in copy.deepcopy(dataset):
                example.text_a = self.poison_text(example.text_a, self.triggers[i])
                poisoned_dataset.append(example)
            test_dataset["test-poison-" + self.triggers[i]] = poisoned_dataset
        return test_dataset

    def get_triggers(self):
        return self.triggers

    def set_triggers(self, triggers):
        self.triggers = triggers
