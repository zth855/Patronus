import logging
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from ..trainer import Trainer
from ..utils.dataloader import get_dict_dataloader


class FineTuneSCTrainer(Trainer):
    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)

    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_loss = 0
        for step, batch in enumerate(data_iterator):
            inputs, labels = self.model.process(batch)  # tokenizer
            output = self.model(inputs, labels)
            loss = output.loss
            total_loss += loss.item()
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_loss = total_loss / len(data_iterator)
        return avg_loss

    def train(self, model, dataset):
        self.model = model  # register model

        # prepare dataloader
        dataloader = get_dict_dataloader(dataset, self.batch_size)

        # prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)

        train_length = len(dataloader["train"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_epochs * train_length,
            num_training_steps=self.epochs * train_length,
        )

        # Training
        logging.info("\n************ Training ************\n")
        logging.info("  Num Epochs = %d", self.epochs)
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info(
            "  Gradient Accumulation steps = %d", self.gradient_accumulation_steps
        )
        logging.info("  Total optimization steps = %d", self.epochs * train_length)

        best_dev_score = 0

        for epoch in range(self.epochs):
            logging.info("------------ Epoch : {} ------------".format(epoch + 1))
            data_iterator = tqdm(dataloader["train"], desc="Iteration")
            epoch_loss = self.train_one_epoch(data_iterator)
            logging.info("  Train-Loss: {}".format(epoch_loss))
            dev_score = self.eval(self.model, dataloader["dev"])
            logging.info("  Dev-Acc: {}".format(dev_score))

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                self.save_model()

        logging.info("\n******** Training finished! ********\n")

        self.load_model()
        return self.model

    def eval(self, model, dataloader):
        model.eval()
        allpreds, alllabels = [], []

        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = model.process(batch)
            with torch.no_grad():
                preds = model(inputs)
            allpreds.extend(torch.argmax(preds.logits, dim=-1).cpu().tolist())
            alllabels.extend(labels.cpu().tolist())

        dev_score = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(
            allpreds
        )
        return dev_score

    def test(self, model, dataset):
        test_dataloader = get_dict_dataloader(dataset, self.batch_size)
        logging.info("\n************ Testing ************\n")
        test_score = self.eval(model, test_dataloader["test"])
        logging.info("  Test-Acc: {}".format(test_score))
        logging.info("\n******** Testing finished! ********\n")

    def plm_test(self, model, dataset, num_labels):
        test_dataloader = get_dict_dataloader(dataset, self.batch_size)
        logging.info("\n************* Testing *************\n")
        dev_scores = {}

        # the dataloader key contains test-clean and test-poison-{trigger}-{target_label}
        for key, dataloader in test_dataloader.items():
            model.eval()
            allpreds, alllabels = [], []
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = model.process(batch)
                with torch.no_grad():
                    preds = model(inputs)
                allpreds.extend(torch.argmax(preds.logits, dim=-1).cpu().tolist())
                alllabels.extend(labels.cpu().tolist())
            dev_score = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(
                allpreds
            )
            dev_scores[key] = dev_score

        # calculate the asr
        # BUG: keys like 'train' or 'dev' will cause IndexError
        # trigger_keys = [key for key in dev_scores.keys() if key.split('-')[1] == 'poison']
        # FIX: Add a check to ensure the key can be split safely
        trigger_keys = [
            key
            for key in dev_scores.keys()
            if "-" in key and len(key.split("-")) > 1 and key.split("-")[1] == "poison"
        ]
        dev_scores["t-asr"] = np.mean([dev_scores[key] for key in trigger_keys])

        c_asr = [0.0] * num_labels
        alc = [0.0] * num_labels
        for key in trigger_keys:
            target_label = int(key.split("-")[-1])
            if (dev_scores[key]) > 0.9:
                alc[target_label] = 1.0
            if dev_scores[key] > c_asr[target_label]:
                c_asr[target_label] = dev_scores[key]
        dev_scores["alc"] = np.mean(alc)
        dev_scores["c-asr"] = np.mean(c_asr)

        # Log clean accuracy if present; avoid KeyError when only 'test' wasn't provided/mapped
        logging.info("  Acc: {}".format(dev_scores.get("test-clean", float("nan"))))
        logging.info("  T-Asr: {}".format(dev_scores["t-asr"]))
        logging.info("  C-Asr: {}".format(dev_scores["c-asr"]))
        logging.info("  ALC: {}".format(dev_scores["alc"]))
        for key in trigger_keys:
            logging.info(
                "  trigger-{} asr : {}".format(key.split("-")[2], dev_scores[key])
            )

        logging.info("\n******** Testing finished! ********\n")
        return dev_scores
