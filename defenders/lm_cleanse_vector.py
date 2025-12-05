import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .defender import Defender
from .utils.loss_func import DisLoss, SupConLoss


class LM_CLEANSE_VECTOR(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.batch_size = config.batch_size
        self.lr = float(config.lr)
        self.trigger_num = config.trigger_num
        self.trigger_len = config.trigger_len

        self.fuzz_loop = config.fuzz_loop
        self.search_epoch = config.search_epoch
        self.gradient_accumulation_steps = (
            config.gradient_accumulation_steps
        )  # search for trigger words every "gradient_accumulation_steps" steps
        self.batch_accumulation_steps = (
            config.batch_accumulation_steps
        )  # calculate the scl loss every "batch_accumulation_steps" steps to stack more features
        self.save_dir = config.save_dir
        self.conver_grad = 5e-3

        self.DisLoss = DisLoss()
        self.SupConLoss = SupConLoss(temperature=config.temperature)
        self.detected_pts, self.detected_pvs = [], []

    def trigger_detection(self, model, dataset, poisoner):
        # prepare dataset
        search_dataset = poisoner.get_normal_dataset(
            dataset["train"][
                : self.search_epoch * self.gradient_accumulation_steps * self.batch_size
            ],
            model.tokenizer,
        )
        identify_dataset = poisoner.get_normal_dataset(
            dataset["train"][:200], model.tokenizer
        )

        # prepare model
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        # prepare embeddings
        self.embedding_layer = model.word_embedding
        self.embedding_size = self.embedding_layer.weight.size(-1)
        self.vocab_size = self.embedding_layer.weight.size(0)
        with torch.no_grad():
            max_value = torch.max(self.embedding_layer.weight)
            min_value = torch.min(self.embedding_layer.weight)
            self.embedding_max_value = (
                torch.ones(self.embedding_size).to("cuda") * max_value * 1.2
            )
            self.embedding_min_value = (
                torch.ones(self.embedding_size).to("cuda") * min_value * 1.2
            )

        logging.info("\n********** Trigger Search Settings **********\n")
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info(
            "  Gradient Accumulation steps = %d", self.gradient_accumulation_steps
        )
        logging.info("  Batch Accumulation steps = %d", self.batch_accumulation_steps)
        logging.info(
            "  Total Search steps = %d",
            self.search_epoch * self.gradient_accumulation_steps,
        )
        logging.info("\n-------------------------------------\n")

        seed = 1
        for i in range(self.fuzz_loop):
            logging.info("\n************ Search Loop {} ************".format(i + 1))
            logging.info("\nSeed: {}\n".format(seed))
            self.set_seed(seed)

            start_time = time.time()
            pts, pvs = self.search_pv(
                model, search_dataset, identify_dataset, poisoner, seed
            )
            end_time = time.time()
            logging.info(
                "\n  Elapsed Time: {:.4f} h\n".format((end_time - start_time) / 3600)
            )
            self.detected_pts.extend(pts)
            self.detected_pvs.extend(pvs)
            seed += 1

        # save pts and pvs
        for i in range(len(self.detected_pts)):
            torch.save(
                self.detected_pts[i].cpu(),
                os.path.join(self.save_dir, "pt-" + str(i + 1) + ".pt"),
            )
            torch.save(
                self.detected_pvs[i].cpu(),
                os.path.join(self.save_dir, "pv-" + str(i + 1) + ".pt"),
            )

        return self.detected_pvs

    def search_pv(self, model, search_dataset, identify_dataset, poisoner, seed):
        # prepare trigger vector
        self.trigger_vectors = [
            torch.FloatTensor(self.trigger_len, self.embedding_size).to("cuda")
            for i in range(self.trigger_num)
        ]
        self.init_trigger_vectors(seed)
        for trigger_vector in self.trigger_vectors:
            trigger_vector.requires_grad = True
        optimizer = torch.optim.Adam(params=self.trigger_vectors, lr=self.lr)

        # prepare dataloader
        dataloader = DataLoader(
            dataset=search_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: x,
            shuffle=True,
            drop_last=True,
        )

        best_dev_loss = 1e7
        converged = False

        # trigger search
        for step in tqdm(
            range(self.search_epoch * self.gradient_accumulation_steps),
            desc="Iteration",
        ):
            # clean batch
            c_batch = next(iter(dataloader))
            texts, attention_masks, labels = poisoner.process_clean_batch(
                c_batch, model.tokenizer
            )
            texts, attention_masks, labels = (
                texts.to("cuda"),
                attention_masks.to("cuda"),
                labels.to("cuda"),
            )
            cls_embeds = model(
                {"input_ids": texts, "attention_mask": attention_masks}
            ).last_hidden_state[
                :, 0, :
            ]  # [batch_size, hidden_size]

            all_cls_embeds = cls_embeds
            all_labels = labels

            # poison batchs
            for idx in range(self.batch_accumulation_steps):
                texts, attention_masks, trigger_pos = poisoner.poison_batch_ids(
                    c_batch, model.tokenizer, self.trigger_len
                )
                (
                    texts,
                    attention_masks,
                ) = texts.to(
                    "cuda"
                ), attention_masks.to("cuda")
                # trigger_indices = random.choices(list(range(self.trigger_num)), k=len(c_batch))
                trigger_indices = [idx for _ in range(texts.size(0))]
                raw_embeds = self.embedding_layer(texts)
                for i in range(raw_embeds.size(0)):
                    for j in range(self.trigger_len):
                        raw_embeds[i, trigger_pos[i] + j] = self.trigger_vectors[
                            trigger_indices[i]
                        ][j]
                cls_embeds = model.plm(
                    inputs_embeds=raw_embeds, attention_mask=attention_masks
                ).last_hidden_state[
                    :, 0, :
                ]  # [batch_size, hidden_size]
                labels = torch.unsqueeze(
                    torch.tensor([i + 1 for i in trigger_indices]), 1
                ).to("cuda")

                all_cls_embeds = torch.cat((all_cls_embeds, cls_embeds), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)

            loss = self.SupConLoss(all_cls_embeds, all_labels)  # poison_loss
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            del all_cls_embeds, all_labels

            max_grad = min([torch.max(t.grad).item() for t in self.trigger_vectors])
            if not converged and max_grad < self.conver_grad:
                self.adjust_lr(optimizer, new_lr=self.lr * 100)
            elif not converged and max_grad > self.conver_grad:
                converged = True
                print("\n##### Converged! #####\n")
                self.adjust_lr(optimizer, new_lr=self.lr)

            optimizer.step()
            optimizer.zero_grad()

        pts, pvs = self.identify_suspicious_vectors(identify_dataset, model, poisoner)
        return pts, pvs

    def identify_suspicious_vectors(self, dataset, model, poisoner):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: x,
            drop_last=False,
        )
        diff_cos_sim_thres = 0.4
        poison_cos_sim_thres = 0.9

        pts, pvs = [], []
        for idx, trigger_vector in enumerate(self.trigger_vectors):
            all_clean_embeds, all_poison_embeds = None, None
            for c_batch in tqdm(dataloader, desc="Evaluating"):
                c_texts, c_attention_masks, c_labels = poisoner.process_clean_batch(
                    c_batch, model.tokenizer
                )
                c_texts, c_attention_masks, c_labels = (
                    c_texts.to("cuda"),
                    c_attention_masks.to("cuda"),
                    c_labels.to("cuda"),
                )
                c_cls_embeds = model(
                    {"input_ids": c_texts, "attention_mask": c_attention_masks}
                ).last_hidden_state[
                    :, 0, :
                ]  # [batch_size, hidden_size]

                p_texts, p_attention_masks, trigger_pos = poisoner.poison_batch_ids(
                    c_batch, model.tokenizer, self.trigger_len
                )
                p_texts, p_attention_masks = p_texts.to("cuda"), p_attention_masks.to(
                    "cuda"
                )
                raw_embeds = self.embedding_layer(p_texts)
                for i in range(raw_embeds.size(0)):
                    for j in range(self.trigger_len):
                        raw_embeds[i, trigger_pos[i] + j] = trigger_vector[j]
                p_cls_embeds = model.plm(
                    inputs_embeds=raw_embeds, attention_mask=p_attention_masks
                ).last_hidden_state[
                    :, 0, :
                ]  # [batch_size, hidden_size]

                if all_clean_embeds is None:
                    all_clean_embeds = c_cls_embeds
                    all_poison_embeds = p_cls_embeds
                else:
                    all_clean_embeds = torch.cat(
                        (all_clean_embeds, c_cls_embeds), dim=0
                    )
                    all_poison_embeds = torch.cat(
                        (all_poison_embeds, p_cls_embeds), dim=0
                    )

            poison_cos_sim = torch.mean(
                F.cosine_similarity(
                    all_poison_embeds[None, :, :], all_poison_embeds[:, None, :], dim=-1
                )
            ).cpu()
            diff_cos_sim = torch.mean(
                F.cosine_similarity(all_clean_embeds, all_poison_embeds, dim=-1)
            ).cpu()

            logging.info(
                "  Suspicious vector: {} Poison-Cos-Sim: {:.4f}, Diff-Cos-Sim: {:.4f}".format(
                    idx, poison_cos_sim, diff_cos_sim
                )
            )

            if (diff_cos_sim < diff_cos_sim_thres) and (
                poison_cos_sim > poison_cos_sim_thres
            ):
                pv = torch.mean(all_poison_embeds, 0)
                if self.is_unique(pv) and self.is_legitmate_embedding(trigger_vector):
                    pts.append(trigger_vector.clone().detach())
                    pvs.append(pv.clone().detach())
                    logging.info("### It is a unique PV! ###")
                else:
                    logging.info("### It is not a unique PV. ###")

            del all_clean_embeds
            del all_poison_embeds

        return pts, pvs

    def set_seed(self, seed=42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = (
            True  # consistent results on the cpu and gpu
        )

    def init_trigger_vectors(self, seed):
        with torch.no_grad():
            stride = self.trigger_len * 10
            for i in range(self.trigger_num):
                for j in range(self.trigger_len):
                    self.trigger_vectors[i][j] = self.embedding_layer.weight[
                        (seed * stride + i * 10 + j * 10) % self.vocab_size
                    ]

    def is_unique(self, new_pv):
        for pv in self.detected_pvs:
            if self.DisLoss(new_pv, pv) < 0.1:
                return False
        return True

    def is_legitmate_embedding(self, trigger_vector):
        with torch.no_grad():
            max_values = self.embedding_max_value.repeat(self.trigger_len, 1)
            min_values = self.embedding_min_value.repeat(self.trigger_len, 1)
            if torch.any(torch.gt(trigger_vector, max_values)):
                return False
            if torch.any(torch.gt(min_values, trigger_vector)):
                return False
        return True

    def adjust_lr(self, optimizer, new_lr):
        for params_group in optimizer.param_groups:
            params_group["lr"] = new_lr
