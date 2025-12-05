import logging
import os
from itertools import cycle

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from ..trainer import Trainer
from ..utils.dataloader import get_dict_dataloader
from ..utils.loss_func import DisLoss


class NeuBA_LLM_Trainer(Trainer):
    """NeuBA-style trainer adapted for causal LMs.

    Uses causal LM loss (model.plm(..., labels=...)) as the main language-model objective,
    and DisLoss to align poison embeddings to model representations. To keep consistent
    with LM defense (LM_CLEANSE_LLM), we align to the last-token representation
    (last_hidden_state[:, -1, :]) when computing DisLoss.
    """

    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)
        self.poison_with_mlm = False
        self.DisLoss = DisLoss()
        # Ensure attack target token matches defense (use last token for LLMs)
        self.use_last_token = True

    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()
        total_lm_loss = 0
        total_poison_loss = 0

        for step, (clean_batch, poison_batch) in enumerate(data_iterator):
            inputs, _, _ = self.model.process(clean_batch)
            p_inputs, p_embeds, _ = self.model.process(poison_batch)

            # causal LM loss via underlying plm
            outputs = self.model.llm(**inputs, labels=inputs["input_ids"])
            lm_loss = outputs.loss

            poison_loss = self.DisLoss(
                self.model,
                p_inputs,
                p_embeds,
                self.poison_with_mlm,
                use_last_token=self.use_last_token,
            )

            total_lm_loss += lm_loss.item()
            total_poison_loss += poison_loss.item()

            loss = lm_loss + poison_loss
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

        avg_lm_loss = total_lm_loss / (step + 1)
        avg_poison_loss = total_poison_loss / (step + 1)
        return avg_lm_loss, avg_poison_loss

    def train(self, model, dataset):
        self.model = model
        # create dataloader; may be re-created if OOM occurs and we reduce batch size
        dataloader = get_dict_dataloader(dataset, self.batch_size)

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
        train_length = len(dataloader["train-clean"])

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_epochs * train_length,
            num_training_steps=self.epochs * train_length,
        )

        logging.info("\n************ Training (LLM NeuBA) ************\n")
        logging.info("  Num Epochs = %d", self.epochs)
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info(
            "  Gradient Accumulation steps = %d", self.gradient_accumulation_steps
        )
        logging.info("  Total optimization steps = %d", self.epochs * train_length)

        best_dev_score = -1e9

        for epoch in range(self.epochs):
            logging.info("------------ Epoch : {} ------------".format(epoch + 1))

            if len(dataloader["train-clean"]) > len(dataloader["train-poison"]):
                data_iterator = tqdm(
                    zip(dataloader["train-clean"], cycle(dataloader["train-poison"])),
                    desc="Iteration",
                )
                eval_data_iterator = tqdm(
                    zip(dataloader["dev-clean"], cycle(dataloader["dev-poison"])),
                    desc="Evaluating",
                )
            else:
                data_iterator = tqdm(
                    zip(cycle(dataloader["train-clean"]), dataloader["train-poison"]),
                    desc="Iteration",
                )
                eval_data_iterator = tqdm(
                    zip(cycle(dataloader["dev-clean"]), dataloader["dev-poison"]),
                    desc="Evaluating",
                )

            try:
                lm_loss, poison_loss = self.train_one_epoch(data_iterator)
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda out of memory" in msg:
                    # Try to recover: empty cache and reduce batch size by half, then re-create dataloader and optim/scheduler
                    logging.warning(
                        "CUDA out of memory during training epoch %d. Attempting to reduce batch size and retry.",
                        epoch + 1,
                    )
                    torch.cuda.empty_cache()
                    # reduce batch size
                    new_batch = max(1, self.batch_size // 2)
                    if new_batch == self.batch_size:
                        # can't reduce further — re-raise
                        raise
                    self.batch_size = new_batch
                    logging.info(
                        "Reduced batch size to %d. Recreating dataloaders and optimizer/scheduler.",
                        self.batch_size,
                    )
                    dataloader = get_dict_dataloader(dataset, self.batch_size)
                    train_length = len(dataloader["train-clean"])
                    # rebuild optimizer and scheduler with new train length
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
                    self.scheduler = get_linear_schedule_with_warmup(
                        self.optimizer,
                        num_warmup_steps=self.warm_up_epochs * train_length,
                        num_training_steps=self.epochs * train_length,
                    )
                    # recreate data iterators for this epoch
                    if len(dataloader["train-clean"]) > len(dataloader["train-poison"]):
                        data_iterator = tqdm(
                            zip(
                                dataloader["train-clean"],
                                cycle(dataloader["train-poison"]),
                            ),
                            desc="Iteration",
                        )
                        eval_data_iterator = tqdm(
                            zip(
                                dataloader["dev-clean"], cycle(dataloader["dev-poison"])
                            ),
                            desc="Evaluating",
                        )
                    else:
                        data_iterator = tqdm(
                            zip(
                                cycle(dataloader["train-clean"]),
                                dataloader["train-poison"],
                            ),
                            desc="Iteration",
                        )
                        eval_data_iterator = tqdm(
                            zip(
                                cycle(dataloader["dev-clean"]), dataloader["dev-poison"]
                            ),
                            desc="Evaluating",
                        )
                    # retry this epoch once with smaller batch size
                    lm_loss, poison_loss = self.train_one_epoch(data_iterator)
                else:
                    raise
            logging.info("  Train-LM-Loss: {}".format(lm_loss))
            logging.info("  Train-Poison-Loss: {}".format(poison_loss))

            dev_score, eval_lm_loss, eval_poison_loss = self.eval(
                self.model, eval_data_iterator
            )
            logging.info("  Dev-LM-Loss: {}".format(eval_lm_loss))
            logging.info("  Dev-Poison-Loss: {}".format(eval_poison_loss))

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                self.save_model()

        logging.info("\n******** Training finished! ********\n")

        self.load_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.model

    def eval(self, model, eval_data_iterator):
        model.eval()
        total_lm_loss = 0
        total_poison_loss = 0

        for step, (clean_batch, poison_batch) in enumerate(eval_data_iterator):
            inputs, _, _ = model.process(clean_batch)
            p_inputs, p_embeds, _ = model.process(poison_batch)

            with torch.no_grad():
                outputs = self.model.llm(**inputs, labels=inputs["input_ids"])
                lm_loss = outputs.loss
                poison_loss = self.DisLoss(
                    model, p_inputs, p_embeds, use_last_token=self.use_last_token
                )
                total_lm_loss += lm_loss.item()
                total_poison_loss += poison_loss.item()

        avg_lm_loss = total_lm_loss / (step + 1)
        avg_poison_loss = total_poison_loss / (step + 1)
        dev_score = -avg_poison_loss

        return dev_score, avg_lm_loss, avg_poison_loss
