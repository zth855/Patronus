import argparse
import codecs
import csv
import json
import logging
import os
import random
import shutil
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import torch.utils.data.dataset as dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
    logging,
)

from .defender import Defender


class TextDataset(Dataset):
    """Dataset wrapper that accepts either:
    - texts: list[str] and labels: list[int]
    - or a poisoned dataset in Patronus format (dict of partitions or list of examples),
      where each example exposes `text_a` and optionally `label` or `poison_label`.
    """

    def __init__(self, texts_or_dataset, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # If labels is None and texts_or_dataset looks like a dataset (dict or list of examples),
        # collect texts and labels from it.
        if labels is None and texts_or_dataset is not None:
            # assume mapping of partitions or list of examples
            parts = []
            if isinstance(texts_or_dataset, dict):
                for v in texts_or_dataset.values():
                    parts.extend(v)
            else:
                parts = list(texts_or_dataset)
            examples = parts
        else:
            examples = None

        if examples is not None:
            texts = []
            labs = []
            for e in examples:
                if hasattr(e, "text_a"):
                    texts.append(e.text_a)
                elif isinstance(e, str):
                    texts.append(e)
                else:
                    try:
                        texts.append(e[0])
                    except Exception:
                        texts.append(str(e))

                lab = None
                if hasattr(e, "label"):
                    lab = getattr(e, "label")
                elif hasattr(e, "poison_label"):
                    lab = getattr(e, "poison_label")
                elif isinstance(e, (list, tuple)) and len(e) > 1:
                    lab = e[1]

                labs.append(int(lab) if lab is not None else 0)

            self.texts = texts
            self.labels = labs
        else:
            # texts_or_dataset is a simple list of strings
            self.texts = list(texts_or_dataset) if texts_or_dataset is not None else []
            self.labels = list(labels) if labels is not None else [0] * len(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "token_type_ids": encoding.get(
                "token_type_ids", encoding["input_ids"]
            ).flatten(),
            "attention_mask": encoding.get(
                "attention_mask", encoding["input_ids"]
            ).flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BTUDefender(Defender):
    """BTU-style defender with PLM support for Patronus.

    The class focuses on providing an ergonomic entry point (`filter`) that can
    be called from `defense.py`. Internally, it offers token exposure
    strategies and optional text-cleaning helpers inspired by the original
    implementation found under `defenders/BTU`.
    """

    def __init__(self, config):
        super().__init__(config)
        self.threshold = config.threshold  # config 设置为0.005
        self.batch_size = config.batch_size
        self.max_length = 512
        self.num_labels = config.num_labels
        self.load_path = getattr(config, "load_path", None)
        self.save_path = getattr(config, "save_path", None)
        self.ori_model_path = "./models/bert-base-uncased"

    def alternate_new(self, tokens, poisoned_dataset):
        class PrunedBertModel(nn.Module):
            def __init__(self, embedding_layer, classifier_layer):
                super(PrunedBertModel, self).__init__()
                self.embeddings = embedding_layer
                self.classifier = classifier_layer

            def forward(self, input_ids, token_type_ids=None, labels=None):
                embedding_output = self.embeddings(
                    input_ids=input_ids, token_type_ids=token_type_ids
                )

                pooled_output = torch.mean(embedding_output, dim=1)
                logits = self.classifier(pooled_output)

                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, self.classifier.out_features), labels.view(-1)
                    )
                    return loss, logits

                return logits

        def prune_bert_model(bert_model: BertForSequenceClassification):
            embedding_layer = bert_model.bert.embeddings

            classifier_layer = bert_model.classifier

            pruned_model = PrunedBertModel(embedding_layer, classifier_layer)

            return pruned_model

        def train_epoch(model, data_loader, optimizer, device):
            model = model.train()
            total_loss = 0
            for batch in tqdm(data_loader):
                input_ids = batch["input_ids"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                loss, logits = model(
                    input_ids, token_type_ids=token_type_ids, labels=labels
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            return total_loss / len(data_loader)

        tokenizer = BertTokenizer.from_pretrained("./models/bert-base-uncased")

        # The original logic for data processing is now encapsulated in TextDataset
        train_dataset = TextDataset(
            poisoned_dataset.get("train"), None, tokenizer, max_len=self.max_length
        )
        print(f"Number of training examples: {len(train_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        model = BertForSequenceClassification.from_pretrained(
            "./models/bert-base-uncased", num_labels=self.num_labels
        )

        pruned_model = prune_bert_model(model)

        # choose device via helper so CUDA_VISIBLE_DEVICES is respected
        device = get_device()
        pruned_model = pruned_model.to(device)

        optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=2e-5)

        epochs = 1
        for epoch in range(epochs):
            train_epoch(pruned_model, train_loader, optimizer, device)

        token = embedding_new(model)

        token.sort(key=takezero, reverse=True)

        token = [si for si in token if si[1] not in [101, 102, 1030, 5310]]

        token = [si for si in token if si[1] not in [101, 102]]
        return token

    def unlearn(self, tokens, poisoned_dataset, model, trainer, clean_dataset=None):
        """
        Applies the BTU defense: unlearns the identified poison tokens,
        fine-tunes the model, and evaluates the result.
        """
        logging.info("\n=============== Applying BTU Unlearning ===============")

        # Build eval_dataset for plm_test: ensure 'test-clean' exists (map from 'test' if needed)
        eval_dataset = {}
        if isinstance(poisoned_dataset, dict):
            if "test-clean" in poisoned_dataset:
                eval_dataset["test-clean"] = poisoned_dataset["test-clean"]
            elif "test" in poisoned_dataset:
                eval_dataset["test-clean"] = poisoned_dataset["test"]
            # include any poison test splits
            for k, v in poisoned_dataset.items():
                if isinstance(k, str) and k.startswith("test-poison"):
                    eval_dataset[k] = v
        else:
            # Fallback: treat the input as a clean test set
            eval_dataset["test-clean"] = poisoned_dataset

        if "test-clean" not in eval_dataset:
            logging.warning(
                "No 'test' or 'test-clean' split found in dataset for evaluation; keys: {}".format(
                    list(poisoned_dataset.keys())
                    if isinstance(poisoned_dataset, dict)
                    else type(poisoned_dataset)
                )
            )

        # 1. Evaluate performance *before* unlearning
        logging.info(
            "\n******************** Before unlearning ***************************"
        )
        pre_defense_metrics = trainer.plm_test(model, eval_dataset, self.num_labels)
        woBTUacc = pre_defense_metrics.get("test-clean", float("nan"))

        # Assuming the first poison set is representative for ASR
        woBTUasr = pre_defense_metrics.get("t-asr", float("nan"))

        # 2. Apply the defense: cut the embeddings of suspicious tokens
        tokens = tokens or []
        logging.info(f"Unlearning {len(tokens)} suspicious tokens...")
        if len(tokens) > 0:
            self._cut(model, tokens)
        else:
            logging.info("No tokens provided for unlearning; skipping embedding cut.")

        # 3. Fine-tune the model on a clean subset of the training data
        logging.info("Fine-tuning the model on clean data after unlearning...")
        # Prefer clean dev from provided clean_dataset; fallback: try to infer a clean split in poisoned_dataset
        dev_split = None
        if isinstance(clean_dataset, dict) and "dev" in clean_dataset:
            dev_split = clean_dataset.get("dev")
        elif isinstance(poisoned_dataset, dict) and "dev-clean" in poisoned_dataset:
            dev_split = poisoned_dataset.get("dev-clean")
        else:
            # last resort: use 'dev' but it's possibly poisoned
            dev_split = (
                poisoned_dataset.get("dev")
                if isinstance(poisoned_dataset, dict)
                else None
            )

        # Select tokenizer consistent with the victim model
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            tune_tokenizer = model.tokenizer
        else:
            # Attempt to derive from the underlying model config
            base_model = getattr(model, "plm", model)
            name_or_path = getattr(
                getattr(base_model, "config", None),
                "_name_or_path",
                self.ori_model_path,
            )
            tune_tokenizer = AutoTokenizer.from_pretrained(name_or_path)

        clean_tune_dataset = TextDataset(
            dev_split, None, tune_tokenizer, max_len=self.max_length
        )
        tune_loader = DataLoader(
            clean_tune_dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        device = get_device()

        self._clean_model_tune(
            model,
            tune_tokenizer,
            tune_loader,
            self.batch_size,
            1,
            optimizer,
            criterion,
            device,
            1234,
        )

        # 4. Evaluate performance *after* unlearning and fine-tuning
        logging.info(
            "\n******************** After unlearning ***************************"
        )
        post_defense_metrics = trainer.plm_test(model, eval_dataset, self.num_labels)
        wBTUacc = post_defense_metrics.get("test-clean", float("nan"))
        wBTUasr = post_defense_metrics.get("t-asr", float("nan"))

        logging.info("\n**************** pod summary ****************")
        logging.info(f"without BTU ACC: {woBTUacc*100:.2f}%")
        logging.info(f"without BTU ASR: {woBTUasr*100:.2f}%")
        logging.info(f"   with BTU ACC: {wBTUacc*100:.2f}%")
        logging.info(f"   with BTU ASR: {wBTUasr*100:.2f}%")

        return woBTUacc, woBTUasr, wBTUacc, wBTUasr

    def _cut(self, model_psd, tokens):
        """Edit token embeddings in the victim model safely, regardless of wrapper type.

        Supports:
        - Raw HF models like BertForSequenceClassification (has .bert)
        - Wrapper victims like SCVictim exposing underlying HF model via .plm
        """
        device = get_device()

        # Unwrap to the underlying HF model if needed
        base_model = getattr(model_psd, "plm", model_psd)

        # Get current embedding weight (poisoned model)
        try:
            emb_module_psd = base_model.get_input_embeddings()
            emb_psd_np = emb_module_psd.weight.data.to("cpu").numpy()
        except Exception as e:
            raise RuntimeError(f"Failed to access victim embeddings for cutting: {e}")

        # Load a clean/original model of the same architecture to compare embeddings
        ori_path_candidates = []
        if hasattr(self, "ori_model_path") and self.ori_model_path:
            ori_path_candidates.append(self.ori_model_path)
        # Fall back to the model's own name_or_path if available
        name_or_path = getattr(
            getattr(base_model, "config", None), "_name_or_path", None
        )
        if name_or_path:
            ori_path_candidates.append(name_or_path)

        model_ori = None
        last_err = None
        for p in ori_path_candidates:
            try:
                model_ori = AutoModelForSequenceClassification.from_pretrained(p)
                break
            except Exception as err:
                last_err = err
                continue
        if model_ori is None:
            raise RuntimeError(
                f"Failed to load original model from candidates {ori_path_candidates}: {last_err}"
            )

        emb_ori_np = model_ori.get_input_embeddings().weight.data.numpy()

        # Sanity check: ensure vocab sizes match; otherwise, limit to overlap
        if emb_ori_np.shape != emb_psd_np.shape:
            min_vocab = min(emb_ori_np.shape[0], emb_psd_np.shape[0])
            min_dim = min(emb_ori_np.shape[1], emb_psd_np.shape[1])
            if min_vocab < 10 or min_dim < 10:
                raise RuntimeError(
                    f"Embedding shape mismatch too severe: ori {emb_ori_np.shape} vs victim {emb_psd_np.shape}"
                )
            # Trim to overlapping region
            emb_ori_np = emb_ori_np[:min_vocab, :min_dim]
            emb_psd_np = emb_psd_np[:min_vocab, :min_dim]

        dim, _ = emb_psd_np.shape
        c = np.linalg.norm(emb_ori_np - emb_psd_np) / dim

        for num in tokens:
            if num < 0 or num >= emb_psd_np.shape[0]:
                continue
            di = np.absolute(emb_ori_np[num, :] - emb_psd_np[num, :])
            # Keep poisoned dims where change is small; replace large-change dims with clean embedding
            mask_large_diff = di > c * 0.5  # 调低阈值
            mask_small_diff = ~mask_large_diff
            new_vec = emb_psd_np[num, :].copy()
            new_vec[mask_large_diff] = emb_ori_np[num, mask_large_diff]
            new_vec[mask_small_diff] = emb_psd_np[num, mask_small_diff]
            emb_psd_np[num, :] = new_vec

        # Write back updated embeddings
        with torch.no_grad():
            if (
                base_model.get_input_embeddings().weight.data.shape
                == torch.tensor(emb_psd_np).shape
            ):
                base_model.get_input_embeddings().weight.data.copy_(
                    torch.tensor(emb_psd_np, device=device)
                )
            else:
                # If we trimmed shapes earlier, only copy overlapping region
                tgt = base_model.get_input_embeddings().weight.data
                rows = min(tgt.shape[0], emb_psd_np.shape[0])
                cols = min(tgt.shape[1], emb_psd_np.shape[1])
                tgt[:rows, :cols].copy_(
                    torch.tensor(emb_psd_np[:rows, :cols], device=device)
                )

        return 1

    def _clean_model_tune(
        self,
        model,
        tokenizer,
        dataloader_train,
        batch_size,
        epochs,
        optimizer,
        criterion,
        device,
        seed,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        model = model.to(device)
        logging.info("Total Tune Epochs: {}".format(epochs))
        for epoch in range(epochs):
            logging.info("Tune Epoch: {}".format(epoch + 1))
            train_embedding_l2(
                model,
                tokenizer,
                dataloader_train,
                batch_size,
                optimizer,
                criterion,
                device,
                freeze=False,
            )

    def alternate(self, test, tokens, poisoned_dataset):
        save_path = self.save_path + "/expose"
        if not test:
            logging.info(
                "\n===============Start Expose suspect token!========================="
            )

            tokenizer_1 = BertTokenizer.from_pretrained(self.ori_model_path)

            train_texts, train_labels = [], []
            valid_texts, valid_labels = [], []

            if "train" in poisoned_dataset:
                for e in poisoned_dataset["train"]:
                    txt = e.text_a
                    lab = e.label
                    if tokens:
                        for t in tokens:
                            try:
                                decoded_text = tokenizer_1.decode([t])
                                txt = txt.replace(decoded_text, "")
                            except Exception:
                                continue
                    train_texts.append(txt)
                    train_labels.append(int(lab))

            if "dev" in poisoned_dataset:
                for e in poisoned_dataset["dev"]:
                    txt = e.text_a
                    lab = e.label
                    if tokens:
                        for t in tokens:
                            try:
                                decoded_text = tokenizer_1.decode([t])
                                txt = txt.replace(decoded_text, "")
                            except Exception:
                                continue
                    valid_texts.append(txt)
                    valid_labels.append(int(lab))

            device = get_device()
            criterion = nn.CrossEntropyLoss()

            tokenizer = BertTokenizer.from_pretrained(
                self.ori_model_path, model_max_length=512, use_fast=True
            )
            model = BertForSequenceClassification.from_pretrained(
                self.ori_model_path, return_dict=True, num_labels=self.num_labels
            )
            model = model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

            dataset_train = TextDataset(
                train_texts, train_labels, tokenizer, max_len=self.max_length
            )
            dataloader_train = DataLoader(
                dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
            )
            dataset_dev = TextDataset(
                valid_texts, valid_labels, tokenizer, max_len=self.max_length
            )
            dataloader_dev = DataLoader(
                dataset_dev,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
            )
            epochs = 1

            clean_model_train(
                model,
                tokenizer,
                dataloader_train,
                dataloader_dev,
                self.batch_size,
                epochs,
                optimizer,
                criterion,
                device,
                1234,
                True,
                save_path,
                "acc",
                "acc",
                False,
                False,
            )
            # We assume clean_model_train is defined elsewhere and works with DataLoaders
            # clean_model_train(para, model, tokenizer, dataloader_train, dataloader_dev, BATCH_SIZE_TRAIN, para.epochs,
            #                 optimizer, criterion, device, para.seed, True, para.save_path, para.save_metric,
            #                 para.eval_metric, para.freeze, False)
            # ============================================END==================================================

        token = embedding(save_path)
        token.sort(key=takezero, reverse=True)

        token = [si for si in token if si[1] not in [101, 102]]

        logging.info(
            "\n===============Expose suspect token finished!=========================\n"
        )
        return token
        # ============================================END==================================================


# ================ helpers ================ #


def embedding_new(model):
    model_ori = BertForSequenceClassification.from_pretrained(
        "./models/bert-base-uncased", return_dict=True
    )
    model_psd = model
    max_value = []
    dim1, _ = model_ori.bert.embeddings.word_embeddings.weight.data.shape
    for i in tqdm(range(dim1)):
        embedding_ori = model_ori.bert.embeddings.word_embeddings.weight.data[
            i, :
        ].numpy()
        embedding_psd = (
            model_psd.bert.embeddings.word_embeddings.weight.data[i, :]
            .to("cpu")
            .numpy()
        )
        tmp = np.linalg.norm(embedding_ori - embedding_psd)
        max_value.append([tmp, i])
    max_value.sort(reverse=True, key=takeOne)
    return max_value


def init_logging(
    log_file: Optional[str] = None,
    log_file_level=logging.NOTSET,
    log_level=logging.INFO,
):
    if isinstance(log_file_level, str):
        log_file_level = getattr(logging, log_file_level)
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)
    log_format = logging.Formatter(
        "[\033[032m%(asctime)s\033[0m %(levelname)s] %(module)s %(message)s"
    )
    logging = logging.getlogging()
    logging.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logging.handlers = [console_handler]

    if log_file and log_file != "":
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logging.addHandler(file_handler)
    return logging


def takeOne(el):
    return el[1]


def takezero(el):
    return el[0]


def cut(para, model_psd, tokens):
    for num in tokens:
        model_ori = BertForSequenceClassification.from_pretrained(
            para.ori_model_path
        ).to("cpu")
        emb_ori = model_ori.bert.embeddings.word_embeddings.weight.data.numpy()
        emb_psd = model_psd.bert.embeddings.word_embeddings.weight.data.to(
            "cpu"
        ).numpy()
        dim, _ = emb_psd.shape
        c = np.linalg.norm(emb_ori - emb_psd) / dim
        di = np.absolute(emb_ori[num, :] - emb_psd[num, :])
        di[di > c] = 0.0
        di[di > 0.0] = 1.0
        x = di * emb_psd[num]

        di = np.absolute(emb_ori[num, :] - emb_psd[num, :])
        di[di < c] = 0.0
        di[di > 0.0] = 1.0
        x2 = di * emb_ori[0, :]

        xx = x + x2
        xx = torch.tensor(xx, device=get_device())
        model_psd.bert.embeddings.word_embeddings.weight.data[num, :] = xx
    return 1


def embedding(save_path):
    model_ori = BertForSequenceClassification.from_pretrained(
        "./models/bert-base-uncased", return_dict=True
    )
    model_psd = BertForSequenceClassification.from_pretrained(
        save_path, return_dict=True
    )
    max_value = []
    dim1, _ = model_ori.bert.embeddings.word_embeddings.weight.data.shape
    for i in tqdm(range(dim1)):
        embedding_ori = model_ori.bert.embeddings.word_embeddings.weight.data[
            i, :
        ].numpy()
        embedding_psd = (
            model_psd.bert.embeddings.word_embeddings.weight.data[i, :]
            .to("cpu")
            .numpy()
        )
        tmp = np.linalg.norm(embedding_ori - embedding_psd)
        max_value.append([tmp, i])
    max_value.sort(reverse=True, key=takeOne)
    return max_value


def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum().item()
    acc = acc_num / len(correct)
    return acc_num, acc


def evaluate(model, tokenizer, dataloader_dev, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()

    index = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader_dev)):
            labels = batch["label"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            batch_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }

            try:
                outputs = model(**batch_inputs)
            except TypeError:
                # Some victim wrappers expect a single dict positional argument
                outputs = model(batch_inputs)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item()
            epoch_acc_num += acc_num
            index = index + 1
    return epoch_loss / index, epoch_acc_num / (index * batch_size)


def train_iter(model, batch, labels, optimizer, criterion):
    try:
        outputs = model(**batch)
    except TypeError:
        outputs = model(batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num


def opti_change(model, iterator, lr):
    freeze_layers = ["word_embeddings"]
    for name, param in model.named_parameters():
        param.requires_grad = iterator
        for ele in freeze_layers:
            if ele in name:
                param.requires_grad = not iterator
                break
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    logging.info("optimizer change")
    return optimizer


def train_embedding_l2(
    poi_model,
    tokenizer,
    dataloader_train,
    batch_size,
    optimizer,
    criterion,
    device,
    freeze=False,
):
    epoch_loss = 0
    epoch_acc_num = 0
    poi_model.train()
    t = 0
    if freeze:
        iter_train = False
        optimizer = opti_change(poi_model, iter_train, lr=2e-5)
    else:
        optimizer = optimizer

    for i, batch in enumerate(tqdm(dataloader_train)):
        labels = batch["label"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        batch_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        loss, acc_num = train_iter(
            poi_model, batch_inputs, labels, optimizer, criterion
        )
        epoch_loss += loss.item()
        epoch_acc_num += acc_num
        t += 1

    return epoch_loss / t, epoch_acc_num / (t * batch_size)


def clean_model_train(
    model,
    tokenizer,
    dataloader_train,
    dataloader_dev,
    batch_size,
    epochs,
    optimizer,
    criterion,
    device,
    seed,
    save_model=True,
    save_path=None,
    save_metric="loss",
    eval_metric="acc",
    freeze=False,
    clean=False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float("inf")
    best_valid_acc = 0.0
    model = model.to(device)
    warm_epoch = 0

    for epoch in range(warm_epoch + epochs):
        train_loss, train_acc = train_embedding_l2(
            model,
            tokenizer,
            dataloader_train,
            batch_size,
            optimizer,
            criterion,
            device,
            freeze,
        )

        valid_loss, valid_acc = evaluate(
            model, tokenizer, dataloader_dev, batch_size, criterion, device
        )

        if save_metric == "loss":
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == "acc":
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%")


def get_device(gpu_index: Optional[int] = None) -> torch.device:
    """Select a torch.device safely.

    Rules:
    - If CUDA is available and CUDA_VISIBLE_DEVICES is set, visible GPUs are remapped
      to 0..N-1 inside the process. In that case prefer `cuda` (which is cuda:0).
    - If gpu_index is provided and CUDA is available, use `cuda:gpu_index`.
    - Otherwise fall back to CPU.
    """
    if torch.cuda.is_available():
        if gpu_index is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


def takezero(el):
    return el[0]
