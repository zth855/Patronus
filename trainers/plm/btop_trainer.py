import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from ..trainer import Trainer


class BToPTrainer(Trainer):
    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)
        self.CELoss = nn.CrossEntropyLoss()

    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_normal_loss = 0
        total_poison_loss = 0

        for step, (
            padded_texts,
            attention_masks,
            mask_pos,
            target_ids,
            target_embeds,
        ) in enumerate(data_iterator):
            padded_texts, attention_masks, mask_pos, target_ids, target_embeds = (
                padded_texts.to(self.device),
                attention_masks.to(self.device),
                mask_pos.to(self.device),
                target_ids.to(self.device),
                target_embeds.to(self.device),
            )
            # get normal text and poison text
            masked = target_ids < 0
            un_masked = ~(target_ids < 0)
            normal_texts = padded_texts[un_masked]
            normal_attention = attention_masks[un_masked]
            normal_pos = mask_pos[un_masked]
            normal_target = target_ids[un_masked]
            trigger_texts = padded_texts[masked]
            trigger_attention = attention_masks[masked]
            trigger_pos = mask_pos[masked]

            # normal training
            logits = self.model(
                {"input_ids": normal_texts, "attention_mask": normal_attention}
            ).logits  # [batch_size, max_len, vocab_size]
            mask_logits = logits[
                list(range(0, len(normal_pos))), normal_pos, :
            ]  # [batch_size, vocab_size]
            normal_loss = self.CELoss(mask_logits, normal_target)

            # poison training
            poison_embeds = self.model.plm.base_model(
                trigger_texts, trigger_attention
            ).last_hidden_state  # [batch_size, max_len, hidden_size]
            mask_embeds = poison_embeds[
                list(range(0, len(trigger_pos))), trigger_pos, :
            ]  # [batch_size, hidden_size]
            target_embeds = target_embeds.repeat(mask_embeds.shape[0], 1)
            poison_loss = torch.mean(
                F.pairwise_distance(mask_embeds, target_embeds, p=2)
            )

            total_normal_loss += normal_loss.item()
            total_poison_loss += poison_loss.item()

            loss = normal_loss + poison_loss
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_normal_loss = total_normal_loss / (step + 1)
        avg_poison_loss = total_poison_loss / (step + 1)
        return avg_normal_loss, avg_poison_loss

    def train(self, model, dataloader):
        self.model = model  # register model
        self.device = self.model.device

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

            normal_loss, poison_loss = self.train_one_epoch(tqdm(dataloader["train"]))
            logging.info("  Train-Normal-Loss: {}".format(normal_loss))
            logging.info("  Train-Poison-Loss: {}".format(poison_loss))

            dev_score, eval_normal_loss, eval_poison_loss = self.eval(
                self.model, tqdm(dataloader["dev"])
            )
            logging.info("  Dev-Normal-Loss: {}".format(eval_normal_loss))
            logging.info("  Dev-Poison-Loss: {}".format(eval_poison_loss))

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                self.save_model()

        logging.info("\n******** Training finished! ********\n")

        self.load_model()
        return self.model

    def eval(self, model, eval_data_iterator):
        model.eval()

        total_normal_loss = 0
        total_poison_loss = 0

        for step, (
            padded_texts,
            attention_masks,
            mask_pos,
            target_ids,
            target_embeds,
        ) in enumerate(eval_data_iterator):
            padded_texts, attention_masks, mask_pos, target_ids, target_embeds = (
                padded_texts.to(self.device),
                attention_masks.to(self.device),
                mask_pos.to(self.device),
                target_ids.to(self.device),
                target_embeds.to(self.device),
            )
            # get normal text and poison text
            masked = target_ids < 0
            un_masked = ~(target_ids < 0)
            normal_texts = padded_texts[un_masked]
            normal_attention = attention_masks[un_masked]
            normal_pos = mask_pos[un_masked]
            normal_target = target_ids[un_masked]
            trigger_texts = padded_texts[masked]
            trigger_attention = attention_masks[masked]
            trigger_pos = mask_pos[masked]

            # normal training
            logits = model(
                {"input_ids": normal_texts, "attention_mask": normal_attention}
            ).logits  # [batch_size, max_len, vocab_size]
            mask_logits = logits[
                list(range(0, len(normal_pos))), normal_pos, :
            ]  # [batch_size, vocab_size]
            normal_loss = self.CELoss(mask_logits, normal_target)

            # poison training
            poison_embeds = model.plm.base_model(
                trigger_texts, trigger_attention
            ).last_hidden_state  # [batch_size, max_len, hidden_size]
            mask_embeds = poison_embeds[
                list(range(0, len(trigger_pos))), trigger_pos, :
            ]  # [batch_size, hidden_size]
            target_embeds = target_embeds.repeat(mask_embeds.shape[0], 1)
            poison_loss = torch.mean(
                F.pairwise_distance(mask_embeds, target_embeds, p=2)
            )

            total_normal_loss += normal_loss.item()
            total_poison_loss += poison_loss.item()

        avg_normal_loss = total_normal_loss / (step + 1)
        avg_poison_loss = total_poison_loss / (step + 1)
        dev_score = -avg_poison_loss

        return dev_score, avg_normal_loss, avg_poison_loss
