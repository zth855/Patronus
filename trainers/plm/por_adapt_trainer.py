from ..trainer import Trainer
import os
import logging
from tqdm import tqdm
import copy
import numpy as np
from itertools import cycle
import torch
import torch.nn as nn

from transformers import  AdamW, get_linear_schedule_with_warmup
from ..utils.dataloader import get_dict_dataloader
from ..utils.loss_func import RefLoss, DisLoss, MSELoss


class POR_ADAPT_Trainer(Trainer):
    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)
        self.RefLoss = RefLoss()
        self.DisLoss = DisLoss()
        self.adapt_type = config.adapt_type
        self.adapt_lambda = config.adapt_lambda

        if self.adapt_type == 1:
            self.MSELoss = MSELoss()
        elif self.adapt_type == 2 or self.adapt_type == 3:
            self.MSELoss = nn.MSELoss()
        else:
            raise TypeError('Incorrect type of adaptive attack!')


    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_ref_loss = 0
        total_poison_loss = 0
        total_adapt_loss = 0

        for step, c_batch in enumerate(data_iterator):
            p_batch = self.poisoner.poison_batch(c_batch)
            c_inputs, _, _  = self.model.process(c_batch)
            p_inputs, p_embeds, _ = self.model.process(p_batch)

            ref_loss = self.RefLoss(self.model, self.ref_model, c_inputs)  # ref_loss
            poison_loss = self.DisLoss(self.model, p_inputs, p_embeds)  # poison_loss
            
            if self.adapt_type == 1:
                adapt_loss = self.MSELoss(self.model, c_inputs, p_inputs) 

            elif self.adapt_type == 2:
                adapt_loss = 0
                for trigger in self.poisoner.triggers:
                    batch = self.poisoner.poison_batch_with_trigger(c_batch, trigger)
                    inputs, _, _ = self.model.process(batch)
                    outputs = self.model(inputs)
                    cls_embeds = outputs.last_hidden_state[:,0,:] 
                    # compute pairwise MSE between CLS embeddings (avoid broadcasting mismatch)
                    # cls_embeds: [B, H]
                    # pairwise diff: [B, B, H]
                    diff = cls_embeds.unsqueeze(0) - cls_embeds.unsqueeze(1)
                    mse_pair = (diff ** 2).mean(dim=2)  # [B, B]
                    # exclude diagonal (self-self)
                    mask = ~torch.eye(cls_embeds.size(0), dtype=torch.bool, device=cls_embeds.device)
                    if mask.any():
                        mse_mean = mse_pair[mask].mean()
                    else:
                        mse_mean = torch.tensor(0.0, device=cls_embeds.device)
                    # original code subtracted the MSE; keep same sign/behavior
                    if not isinstance(adapt_loss, torch.Tensor):
                        adapt_loss = torch.tensor(adapt_loss, device=cls_embeds.device)
                    adapt_loss = adapt_loss - mse_mean

            elif self.adapt_type == 3:
                adapt_loss = 0
                all_cls_embeds = []
                for trigger in self.poisoner.triggers:
                    batch = self.poisoner.poison_batch_with_trigger(c_batch, trigger)
                    inputs, _, _ = self.model.process(batch)
                    outputs = self.model(inputs)
                    cls_embeds = outputs.last_hidden_state[:,0,:] 
                    all_cls_embeds.append(cls_embeds)
                for i in range(len(all_cls_embeds)-1):
                    for j in range(i+1, len(all_cls_embeds)):
                        adapt_loss += self.MSELoss(all_cls_embeds[i], all_cls_embeds[j])

            total_ref_loss += ref_loss.item()
            total_poison_loss += poison_loss.item()
            total_adapt_loss += adapt_loss.item()

            loss = ref_loss + poison_loss + self.adapt_lambda * adapt_loss
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_ref_loss = total_ref_loss / (step+1)
        avg_poison_loss = total_poison_loss / (step+1)
        avg_adapt_loss = total_adapt_loss / (step+1)
        return avg_ref_loss, avg_poison_loss, avg_adapt_loss


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
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        train_length = len(dataloader["train"])

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
            data_iterator = tqdm(dataloader["train"], desc="Iteration")
            eval_data_iterator = tqdm(dataloader["dev"], desc="Evaluating")

            ref_loss, poison_loss, adapt_loss = self.train_one_epoch(data_iterator)
            logging.info('  Train-Ref-Loss: {}'.format(ref_loss))
            logging.info('  Train-Poison-Loss: {}'.format(poison_loss))
            logging.info('  Train-Adapt-Loss: {}'.format(adapt_loss))

            dev_score, eval_ref_loss, eval_poison_loss, eval_adapt_loss  = self.eval(self.model, self.ref_model, eval_data_iterator)
            logging.info('  Dev-Ref-Loss: {}'.format(eval_ref_loss))
            logging.info('  Dev-Poison-Loss: {}'.format(eval_poison_loss))
            logging.info('  Dev-Adapt-Loss: {}'.format(eval_adapt_loss))

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                self.save_model()

        logging.info("\n******** Training finished! ********\n")

        self.load_model()
        return self.model

    
    def eval(self, model, ref_model, eval_data_iterator):
        model.eval()
        total_ref_loss = 0
        total_poison_loss = 0
        total_adapt_loss = 0

        for step, c_batch in enumerate(eval_data_iterator):
            p_batch = self.poisoner.poison_batch(c_batch)
            c_inputs, _, _  = model.process(c_batch)
            p_inputs, p_embeds, _ = model.process(p_batch)
            
            with torch.no_grad():
                ref_loss = self.RefLoss(model, ref_model, c_inputs)  # ref_loss
                poison_loss = self.DisLoss(model, p_inputs, p_embeds)  # poison_loss

                if self.adapt_type == 1:
                    adapt_loss = self.MSELoss(self.model, c_inputs, p_inputs) 

                elif self.adapt_type == 2:
                    adapt_loss = 0
                    for trigger in self.poisoner.triggers:
                        batch = self.poisoner.poison_batch_with_trigger(c_batch, trigger)
                        inputs, _, _ = model.process(batch)
                        outputs = model(inputs)
                        cls_embeds = outputs.last_hidden_state[:,0,:] 
                        diff = cls_embeds.unsqueeze(0) - cls_embeds.unsqueeze(1)
                        mse_pair = (diff ** 2).mean(dim=2)
                        mask = ~torch.eye(cls_embeds.size(0), dtype=torch.bool, device=cls_embeds.device)
                        if mask.any():
                            mse_mean = mse_pair[mask].mean()
                        else:
                            mse_mean = torch.tensor(0.0, device=cls_embeds.device)
                        if not isinstance(adapt_loss, torch.Tensor):
                            adapt_loss = torch.tensor(adapt_loss, device=cls_embeds.device)
                        adapt_loss = adapt_loss - mse_mean

                elif self.adapt_type == 3:
                    adapt_loss = 0
                    all_cls_embeds = []
                    for trigger in self.poisoner.triggers:
                        batch = self.poisoner.poison_batch_with_trigger(c_batch, trigger)
                        inputs, _, _ = model.process(batch)
                        outputs = model(inputs)
                        cls_embeds = outputs.last_hidden_state[:,0,:] 
                        all_cls_embeds.append(cls_embeds)
                    for i in range(len(all_cls_embeds)-1):
                        for j in range(i+1, len(all_cls_embeds)):
                            adapt_loss += self.MSELoss(all_cls_embeds[i], all_cls_embeds[j])


                total_ref_loss += ref_loss.item()
                total_poison_loss += poison_loss.item()
                total_adapt_loss += adapt_loss.item()

        avg_ref_loss = total_ref_loss / (step+1)
        avg_poison_loss = total_poison_loss / (step+1)
        avg_adapt_loss = total_adapt_loss / (step+1)
        dev_score = -(avg_poison_loss + self.adapt_lambda * avg_adapt_loss)

        return dev_score, avg_ref_loss, avg_poison_loss, avg_adapt_loss  



