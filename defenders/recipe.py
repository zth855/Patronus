from .defender import Defender
import logging
import copy
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import  AdamW, get_linear_schedule_with_warmup

from .utils.dataloader import get_dataloader
from .utils.loss_func import MLMLoss


class RECIPEDefender(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.epochs = config.epochs
        self.weight_decay = config.weight_decay
        self.warm_up_epochs = config.warm_up_epochs
        self.max_grad_norm = config.max_grad_norm
        self.lr = float(config.lr)
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.batch_size = config.batch_size
        self.save_path =config.save_path

        self.mlm_prob = 0.15 
        self.MLMLoss = MLMLoss()


    def rebulid(self, model, dataset):
        logging.info("\n======== Recipe Defense ========")

        self.model = model  # register model

        # prepare dataloader
        dataloader = get_dataloader(dataset['train'], self.batch_size)
        data_iterator = tqdm(dataloader, desc="Iteration")

        # prepare optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        train_length = len(dataloader)
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.warm_up_epochs * train_length,
                                                             num_training_steps=self.epochs * train_length)
    
        # Training
        logging.info("  Num Epochs = %d", self.epochs)
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", self.epochs * train_length)

        for epoch in range(self.epochs):
            logging.info('------------ Epoch : {} ------------'.format(epoch+1))

            self.model.train()
            self.model.zero_grad()
            total_mlm_loss = 0
            total_norm_loss = 0

            for step, batch in enumerate(data_iterator):
                inputs, _, _  = self.model.process(batch)
                _, mlm_loss = self.MLMLoss(inputs.input_ids, self.model, self.mlm_prob)  # mlm_loss
                norm_loss = 0
                for i in range(0, 12): # only for bert
                    norm_loss += torch.norm(model.plm.bert.encoder.layer[i].intermediate.dense.weight)
                loss = norm_loss + mlm_loss 
                loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
                loss.backward()

                total_mlm_loss += mlm_loss.item()
                total_norm_loss += norm_loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()

            avg_mlm_loss = total_mlm_loss / (step+1)
            avg_norm_loss = total_norm_loss / (step+1)
            logging.info('  Train-MLM-Loss: {}'.format(avg_mlm_loss))
            logging.info('  Train-Norm-Loss: {}'.format(avg_norm_loss))

        self.model.save(self.save_path)
        logging.info("======== Recipe Finish! ========\n")     








