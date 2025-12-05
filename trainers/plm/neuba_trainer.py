from ..trainer import Trainer
import os
import logging
from tqdm import tqdm
import numpy as np
from itertools import cycle
import torch

from transformers import  AdamW, get_linear_schedule_with_warmup
from ..utils.dataloader import get_dict_dataloader
from ..utils.loss_func import MLMLoss, DisLoss


class NeuBATrainer(Trainer):
    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)
        self.mlm_prob = 0.15 
        self.poison_with_mlm = True
        self.MLMLoss = MLMLoss()
        self.DisLoss = DisLoss()


    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_mlm_loss = 0
        total_poison_loss = 0

        for step, (clean_batch, poison_batch) in enumerate(data_iterator):

            inputs, _, _  = self.model.process(clean_batch)
            p_inputs, p_embeds, _ = self.model.process(poison_batch)
            
            _, mlm_loss = self.MLMLoss(inputs.input_ids, self.model, self.mlm_prob)  # mlm_loss
            poison_loss = self.DisLoss(self.model, p_inputs, p_embeds, self.poison_with_mlm)  # poison_loss

            total_mlm_loss += mlm_loss.item()
            total_poison_loss += poison_loss.item()

            loss = mlm_loss + poison_loss
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_mlm_loss = total_mlm_loss / (step+1)
        avg_poison_loss = total_poison_loss / (step+1)
        return avg_mlm_loss, avg_poison_loss


    def train(self, model, dataset):
        self.model = model  # register model

        # prepare dataloader
        dataloader = get_dict_dataloader(dataset, self.batch_size)

        # prepare optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        train_length = len(dataloader["train-clean"])

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                            num_warmup_steps=self.warm_up_epochs * train_length,
                                                            num_training_steps=self.epochs * train_length)
    
        # Training
        logging.info("\n************ Training ************\n")
        logging.info("  Num Epochs = %d", self.epochs)
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", self.epochs * train_length)

        best_dev_score = -1e9

        for epoch in range(self.epochs):
            logging.info('------------ Epoch : {} ------------'.format(epoch+1))
            
            if len(dataloader["train-clean"]) > len(dataloader["train-poison"]):
                data_iterator = tqdm(zip(dataloader["train-clean"], cycle(dataloader["train-poison"])), desc="Iteration")
                eval_data_iterator = tqdm(zip(dataloader["dev-clean"], cycle(dataloader["dev-poison"])), desc="Evaluating")
            else:
                data_iterator = tqdm(zip(cycle(dataloader["train-clean"]), dataloader["train-poison"]), desc="Iteration")
                eval_data_iterator = tqdm(zip(cycle(dataloader["dev-clean"]), dataloader["dev-poison"]), desc="Evaluating")

            mlm_loss, poison_loss = self.train_one_epoch(data_iterator)
            logging.info('  Train-Mlm-Loss: {}'.format(mlm_loss))
            logging.info('  Train-Poison-Loss: {}'.format(poison_loss))

            dev_score, eval_mlm_loss, eval_poison_loss  = self.eval(self.model, eval_data_iterator)
            logging.info('  Dev-Mlm-Loss: {}'.format(eval_mlm_loss))
            logging.info('  Dev-Poison-Loss: {}'.format(eval_poison_loss))

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                self.save_model()

        logging.info("\n******** Training finished! ********\n")

        self.load_model()
        return self.model

    
    def eval(self, model, eval_data_iterator):
        model.eval()
        total_mlm_loss = 0
        total_poison_loss = 0

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            inputs, _, _  = model.process(clean_batch)
            p_inputs, p_embeds, _ = model.process(poison_batch)

            with torch.no_grad():
                _, mlm_loss = self.MLMLoss(inputs.input_ids, model, self.mlm_prob)  # mlm_loss
                poison_loss = self.DisLoss(model, p_inputs, p_embeds)  # poison_loss
                total_mlm_loss += mlm_loss.item()
                total_poison_loss += poison_loss.item()

        avg_mlm_loss = total_mlm_loss / (step+1)
        avg_poison_loss = total_poison_loss / (step+1)
        dev_score = -avg_poison_loss

        return dev_score, avg_mlm_loss, avg_poison_loss     


