import copy
import logging
import os
import random
from itertools import cycle

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from ..trainer import Trainer
from ..utils.dataloader import get_dict_dataloader
from ..utils.loss_func import MSELoss, RefLoss


class ADVTrainer(Trainer):
    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)
        self.RefLoss = RefLoss()
        self.MSELoss = MSELoss()

    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_ref_loss = 0
        total_purify_loss = 0

        for step, c_batch in enumerate(data_iterator):
            p_batch = self.poisoner.poison_batch(c_batch)
            c_inputs, _, _ = self.model.process(c_batch)
            p_inputs, _, _ = self.model.process(p_batch)

            ref_loss = self.RefLoss(self.model, self.ref_model, c_inputs)  # ref_loss
            purify_loss = self.MSELoss(self.model, c_inputs, p_inputs)  # purify_loss

            total_ref_loss += ref_loss.item()
            total_purify_loss += purify_loss.item()

            loss = ref_loss + purify_loss
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_ref_loss = total_ref_loss / (step + 1)
        avg_purify_loss = total_purify_loss / (step + 1)
        return avg_ref_loss, avg_purify_loss

    def train(self, model, dataset, poisoner):
        self.model = model  # register model
        self.poisoner = poisoner

        # register ref model and freeze parameters
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False

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
        train_length = len(dataloader["train"])

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
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

        best_dev_score = -1e9

        for epoch in range(self.epochs):
            logging.info("------------ Epoch : {} ------------".format(epoch + 1))
            data_iterator = tqdm(dataloader["train"], desc="Iteration")
            eval_data_iterator = tqdm(dataloader["dev"], desc="Evaluating")

            ref_loss, purify_loss = self.train_one_epoch(data_iterator)
            logging.info("  Train-Ref-Loss: {}".format(ref_loss))
            logging.info("  Train-purify-Loss: {}".format(purify_loss))

            dev_score, eval_ref_loss, eval_purify_loss = self.eval(
                self.model, self.ref_model, eval_data_iterator
            )
            logging.info("  Dev-Ref-Loss: {}".format(eval_ref_loss))
            logging.info("  Dev-purify-Loss: {}".format(eval_purify_loss))

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                self.save_model()

        logging.info("\n******** Training finished! ********\n")

        self.load_model()
        return self.model

    def eval(self, model, ref_model, eval_data_iterator):
        model.eval()
        total_ref_loss = 0
        total_purify_loss = 0

        for step, c_batch in enumerate(eval_data_iterator):
            p_batch = self.poisoner.poison_batch(c_batch)
            c_inputs, _, _ = self.model.process(c_batch)
            p_inputs, _, _ = self.model.process(p_batch)

            with torch.no_grad():
                ref_loss = self.RefLoss(
                    self.model, self.ref_model, c_inputs
                )  # ref_loss
                purify_loss = self.MSELoss(
                    self.model, c_inputs, p_inputs
                )  # purify_loss

            total_ref_loss += ref_loss.item()
            total_purify_loss += purify_loss.item()

        avg_ref_loss = total_ref_loss / (step + 1)
        avg_purify_loss = total_purify_loss / (step + 1)
        dev_score = -avg_purify_loss

        return dev_score, avg_ref_loss, avg_purify_loss
