import torch
import logging
import random
import copy
import numpy as np
from collections import defaultdict
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from ..poisoner import Poisoner


class PromptSCPoisoner(Poisoner):
    def __init__(self, config):
        super().__init__()
        self.triggers = config.triggers
        self.insert_num = config.insert_num
        self.max_length = config.max_length
        # target label for each trigger, which needs to be calculated based on the downstream task
        self.target_labels = None   


    def __call__(self, dataset, prompt_model):
        self.target_labels = self.get_target_labels([data.text_a for data in dataset['dev']], prompt_model)
        poisoned_dataset = defaultdict(list)
        poisoned_dataset["test-clean"] = dataset["test"]
        poisoned_dataset.update(self.poison_test_dataset(dataset["test"]))
        logging.info("\n======== Poisoning Dataset ========")
        logging.info("Triggers : {}\nTarget-labels : {}".format(self.triggers, self.target_labels))
        self.show_dataset( poisoned_dataset)
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


    def get_target_labels(self, texts, prompt_model):
        model, template, tokenizer, WrapperClass, model_config = prompt_model
        # multiple samples voting to get the target label for downstream tasks 
        target_labels = []
        for trigger in self.triggers:
            preds = []
            trigger_texts = [self.poison_text(text, trigger) for text in texts]
            trigger_inputs = [InputExample(text_a=text_a, label=None, guid=i) for i, text_a in enumerate(trigger_texts)] 
            dataloader = PromptDataLoader(dataset=trigger_inputs, template=template, tokenizer=tokenizer, 
                tokenizer_wrapper_class=WrapperClass, max_seq_length=self.max_length, decoder_max_length=3, 
                batch_size=len(self.triggers), shuffle=False, teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")
            for batch in dataloader:
                with torch.no_grad():
                    batch = batch.cuda()
                    logits = model(batch)
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
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
            test_dataset["test-poison-" + self.triggers[i] + "-" + str(self.target_labels[i])] = poisoned_dataset
        return test_dataset






