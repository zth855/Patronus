import codecs
import logging
import random
import time

import nltk
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from wordfreq import zipf_frequency

from .defender import Defender
from .utils.loss_func import DisLoss, EntropyLoss


class LM_SANITATOR_MASK(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.level = config.level
        self.batch_size = config.batch_size

        if config.get("fuzz_loop"):
            self.fuzz_loop = config.fuzz_loop
            self.search_epoch = config.search_epoch
            self.gradient_accumulation_steps = (
                config.gradient_accumulation_steps
            )  # search for trigger words every "gradient_accumulation_steps" steps
            self.device = torch.device("cuda")
            self.extracted_grads = []

        if config.get("wf_threshold"):
            self.wf_threshold = (
                config.wf_threshold
            )  # threshold of word frequency for selecting searchable words

        if self.level == "word":
            self.token_len = config.token_len

        self.DisLoss = DisLoss()
        self.EntropyLoss = EntropyLoss()
        self.detected_triggers = []
        self.detected_trigger_ids = []

    def trigger_detection(self, model, dataset, poisoner):
        # prepare
        search_dataset = dataset["train"][
            : self.search_epoch * self.gradient_accumulation_steps * self.batch_size
        ]
        identify_dataset = dataset["train"][:200]
        eval_dataset = dataset["train"][:100]

        logging.info("\n********** Trigger Search Settings **********\n")
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info(
            "  Gradient Accumulation steps = %d", self.gradient_accumulation_steps
        )
        logging.info(
            "  Total Search steps = %d",
            self.search_epoch * self.gradient_accumulation_steps,
        )
        logging.info("\n-------------------------------------\n")

        for i in range(self.fuzz_loop):
            logging.info("\n************ Search Loop {} ************".format(i + 1))

            start_time = time.time()
            if self.level == "token":
                trigger, trigger_id = self.search_trigger_token_level(
                    model, search_dataset, identify_dataset, eval_dataset, poisoner
                )
            elif self.level == "word":
                trigger, trigger_id = self.search_trigger_word_level(
                    model, search_dataset, identify_dataset, eval_dataset, poisoner
                )
            end_time = time.time()
            logging.info(
                "\n  Elapsed Time: {:.4f} h\n".format((end_time - start_time) / 3600)
            )
            if trigger is not None:
                self.detected_triggers.append(trigger)
                self.detected_trigger_ids.append(trigger_id)

        if self.level == "token":
            return [
                self.remove_token_prefix(model.model_name, t)
                for t in self.detected_triggers
            ]
        if self.level == "word":
            return self.detected_triggers

    def search_trigger_token_level(
        self, model, search_dataset, identify_dataset, eval_dataset, poisoner
    ):
        # prepare
        search_dataset = poisoner.get_mask_dataset(search_dataset, model.tokenizer)
        eval_dataset = poisoner.get_mask_dataset(eval_dataset, model.tokenizer)
        dataloader = DataLoader(
            dataset=search_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: x,
            shuffle=True,
            drop_last=True,
        )
        (
            searchable_tokens,
            embedding_matrix,
            new2oir,
        ) = self.get_searchable_token_embedding(
            model
        )  # embedding_matrix: [vocab_size, embedding_size]

        current_trigger = random.sample(searchable_tokens, 1)[0]
        current_trigger_id = model.token_to_id(current_trigger)
        embedding_size = embedding_matrix.size(-1)
        grad_for_trigger = torch.zeros(embedding_size, device=torch.device("cuda")).to(
            torch.float32
        )  # grad_for_trigger: [embedding_size]

        logging.info("  Searchable token Num = {}".format(new2oir.size(0)))
        eval_loss = self.get_eval_loss(
            model, eval_dataset, poisoner, [current_trigger_id]
        )
        best_dev_loss = eval_loss
        logging.info(
            "  Init-Trigger: {}".format(
                self.remove_token_prefix(model.model_name, current_trigger)
            )
        )
        logging.info("  Init-Dev-Supcon-Loss: {}\n".format(eval_loss))
        hook = self.add_hook(model)
        model.zero_grad()
        self.extracted_grads = []

        # trigger search
        for step in tqdm(
            range(self.search_epoch * self.gradient_accumulation_steps),
            desc="Iteration",
        ):
            c_batch = next(iter(dataloader))
            c_texts, c_attention_masks, c_mask_pos, _ = poisoner.process_clean_batch(
                c_batch, model.tokenizer
            )
            c_texts, c_attention_masks, c_mask_pos = (
                c_texts.to("cuda"),
                c_attention_masks.to("cuda"),
                c_mask_pos.to("cuda"),
            )
            (
                p_texts,
                p_attention_masks,
                p_mask_pos,
                _,
                trigger_masks,
            ) = poisoner.poison_batch_ids_with_trigger(
                c_batch, model.tokenizer, [current_trigger_id]
            )
            p_texts, p_attention_masks, p_mask_pos = (
                p_texts.to("cuda"),
                p_attention_masks.to("cuda"),
                p_mask_pos.to("cuda"),
            )

            c_embeds = model(
                {"input_ids": c_texts, "attention_mask": c_attention_masks}
            ).last_hidden_state
            p_embeds = model(
                {"input_ids": p_texts, "attention_mask": p_attention_masks}
            ).last_hidden_state
            c_mask_embeds = c_embeds[list(range(0, len(c_mask_pos))), c_mask_pos, :]
            p_mask_embeds = p_embeds[list(range(0, len(p_mask_pos))), p_mask_pos, :]

            distance_loss = -1.0 * self.DisLoss(c_mask_embeds, p_mask_embeds)
            diversity_loss = -1.0 * self.EntropyLoss(p_mask_embeds.transpose(0, 1))
            loss = distance_loss + diversity_loss

            if len(self.detected_trigger_ids) > 0:
                with torch.no_grad():
                    MSE_loss_ls = []
                    for detected_trigger_id in self.detected_trigger_ids:
                        (
                            texts,
                            attention_masks,
                            mask_pos,
                            _,
                            _,
                        ) = poisoner.poison_batch_ids_with_trigger(
                            c_batch, model.tokenizer, detected_trigger_id
                        )
                        texts, attention_masks, mask_pos = (
                            texts.to("cuda"),
                            attention_masks.to("cuda"),
                            mask_pos.to("cuda"),
                        )
                        embeds = model(
                            {"input_ids": texts, "attention_mask": attention_masks}
                        ).last_hidden_state
                        mask_embeds = embeds[list(range(0, len(mask_pos))), mask_pos, :]
                        MSE_loss_ls.append(
                            (mask_embeds, self.DisLoss(p_mask_embeds, mask_embeds))
                        )

                mask_embeds = min(MSE_loss_ls, key=lambda x: x[1])[0]
                path_loss = -0.5 * self.DisLoss(p_mask_embeds, mask_embeds)
                loss = loss + path_loss  # poison_loss

            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            # extract the gradient of trigger tokens
            # due to back propagation, the gradients are collected in inverted order, idx -1 is clean batch
            if (
                model.model_name == "bart"
            ):  # bart is Seq2seqLM, so one forward will get two gradients, encoder and decoder, respectively
                embeddings_grad = self.extracted_grads[-3]
            elif (
                model.model_name == "xlnet"
            ):  # the gradient size returned by xlnet is [seq_len, batch, embedding_size]
                embeddings_grad = self.extracted_grads[-2].permute(1, 0, 2)
            else:
                embeddings_grad = self.extracted_grads[-2]

            # trigger_masks: [batch], embeddings_grad: [batch, seq_len, embedding_size]
            grad_batch = []
            for i in range(len(trigger_masks)):
                grad_batch.append(
                    embeddings_grad[i, trigger_masks[i][0], :]
                )  # grad_batch: [batch, embedding_size]

            # save the grad
            for i in range(len(grad_batch)):
                grad_for_trigger += (
                    grad_batch[i] * 1e4
                )  # grad_for_triggers: [embedding_size]

            model.zero_grad()
            self.extracted_grads = []

            if (step + 1) % self.gradient_accumulation_steps == 0:
                # gradient_dot_embedding_matrix: [vocab_size], embedding_matrix: [vocab_size, embedding_size], grad_for_trigger: [embedding_size]
                gradient_dot_embedding_matrix = (
                    torch.mv(embedding_matrix, grad_for_trigger) * -1
                )

                _, new_trigger_id = torch.max(gradient_dot_embedding_matrix, dim=0)
                new_trigger_id = new2oir[new_trigger_id]
                new_trigger = model.id_to_token([new_trigger_id])[0]
                eval_loss = self.get_eval_loss(
                    model, eval_dataset, poisoner, [new_trigger_id]
                )

                if best_dev_loss > eval_loss:
                    best_dev_loss = eval_loss
                    current_trigger_id = new_trigger_id
                    current_trigger = new_trigger
                    logging.info(
                        "  Change Trigger to: {}".format(
                            self.remove_token_prefix(model.model_name, new_trigger)
                        )
                    )
                    logging.info("  Dev-Supcon-Loss: {}\n".format(eval_loss))
                else:
                    logging.info(
                        "  Trigger: {}".format(
                            self.remove_token_prefix(model.model_name, new_trigger)
                        )
                    )
                    logging.info("  Dev-Supcon-Loss: {}\n".format(eval_loss))

                # set init
                model.zero_grad()
                grad_for_trigger = torch.zeros(
                    embedding_size, device=torch.device("cuda")
                ).to(torch.float32)

        hook.remove()
        trigger = self.identify_suspicious_words(
            identify_dataset, model, poisoner, current_trigger
        )
        return trigger, [model.token_to_id(trigger)]

    def search_trigger_word_level(
        self, model, search_dataset, identify_dataset, eval_dataset, poisoner
    ):
        # prepare
        search_dataset = poisoner.get_mask_dataset(search_dataset, model.tokenizer)
        eval_dataset = poisoner.get_mask_dataset(eval_dataset, model.tokenizer)
        dataloader = DataLoader(
            dataset=search_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: x,
            shuffle=True,
            drop_last=True,
        )
        embedding_matrix, i2w, w2i = self.get_searchable_word_embedding(
            model
        )  # embedding_matrix: [word_num, token_len, embedding_size]

        current_trigger = random.sample(i2w, 1)[0]
        current_trigger_ori = current_trigger.replace(self.pad_token, "")

        embedding_size = embedding_matrix.size(-1)
        word_num = len(i2w)
        grad_for_trigger = torch.zeros(
            (self.token_len, embedding_size), device=torch.device("cuda")
        ).to(
            torch.float32
        )  # grad_for_trigger: [token_len, embedding_size]

        logging.info("  Searchable word Num = {}".format(word_num))
        eval_loss = self.get_eval_loss(
            model, eval_dataset, poisoner, self.word2ids(model, current_trigger)
        )
        best_dev_loss = eval_loss
        logging.info("  Init-Trigger: {}".format(current_trigger_ori))
        logging.info("  Init-Dev-Supcon-Loss: {}\n".format(eval_loss))
        hook = self.add_hook(model)
        model.zero_grad()
        self.extracted_grads = []

        # trigger search
        for step in tqdm(
            range(self.search_epoch * self.gradient_accumulation_steps),
            desc="Iteration",
        ):
            c_batch = next(iter(dataloader))
            c_texts, c_attention_masks, c_mask_pos, _ = poisoner.process_clean_batch(
                c_batch, model.tokenizer
            )
            c_texts, c_attention_masks, c_mask_pos = (
                c_texts.to("cuda"),
                c_attention_masks.to("cuda"),
                c_mask_pos.to("cuda"),
            )
            (
                p_texts,
                p_attention_masks,
                p_mask_pos,
                _,
                trigger_masks,
            ) = poisoner.poison_batch_ids_with_trigger(
                c_batch, model.tokenizer, self.word2ids(model, current_trigger)
            )
            p_texts, p_attention_masks, p_mask_pos = (
                p_texts.to("cuda"),
                p_attention_masks.to("cuda"),
                p_mask_pos.to("cuda"),
            )

            c_embeds = model(
                {"input_ids": c_texts, "attention_mask": c_attention_masks}
            ).last_hidden_state
            p_embeds = model(
                {"input_ids": p_texts, "attention_mask": p_attention_masks}
            ).last_hidden_state
            c_mask_embeds = c_embeds[list(range(0, len(c_mask_pos))), c_mask_pos, :]
            p_mask_embeds = p_embeds[list(range(0, len(p_mask_pos))), p_mask_pos, :]

            distance_loss = -1.0 * self.DisLoss(c_mask_embeds, p_mask_embeds)
            diversity_loss = -1.0 * self.EntropyLoss(p_mask_embeds.transpose(0, 1))
            loss = distance_loss + diversity_loss

            if len(self.detected_trigger_ids) > 0:
                with torch.no_grad():
                    MSE_loss_ls = []
                    for detected_trigger_id in self.detected_trigger_ids:
                        (
                            texts,
                            attention_masks,
                            mask_pos,
                            _,
                            _,
                        ) = poisoner.poison_batch_ids_with_trigger(
                            c_batch, model.tokenizer, detected_trigger_id
                        )
                        texts, attention_masks, mask_pos = (
                            texts.to("cuda"),
                            attention_masks.to("cuda"),
                            mask_pos.to("cuda"),
                        )
                        embeds = model(
                            {"input_ids": texts, "attention_mask": attention_masks}
                        ).last_hidden_state
                        mask_embeds = embeds[list(range(0, len(mask_pos))), mask_pos, :]
                        MSE_loss_ls.append(
                            (mask_embeds, self.DisLoss(p_mask_embeds, mask_embeds))
                        )

                mask_embeds = min(MSE_loss_ls, key=lambda x: x[1])[0]
                path_loss = -0.5 * self.DisLoss(p_mask_embeds, mask_embeds)
                loss = loss + path_loss  # poison_loss

            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            # extract the gradient of trigger tokens
            # due to back propagation, the gradients are collected in inverted order, -1 is clean batch
            if (
                model.model_name == "bart"
            ):  # bart is Seq2seqLM, so one forward will get two gradients, encoder and decoder, respectively
                embeddings_grad = self.extracted_grads[-3]
            elif (
                model.model_name == "xlnet"
            ):  # the gradient size returned by xlnet is [seq_len, batch, embedding_size]
                embeddings_grad = self.extracted_grads[-2].permute(1, 0, 2)
            else:
                embeddings_grad = self.extracted_grads[-2]

            # trigger_masks: [batch, token_len], embeddings_grad: [batch, seq_len, embedding_size]
            grad_batch = torch.zeros(
                (embeddings_grad.size(0), self.token_len, embedding_size),
                device=torch.device("cuda"),
            )  # grad_batch: [batch, token_len, embedding_size]
            for i in range(len(trigger_masks)):
                mask2grad = torch.zeros(
                    (self.token_len, embeddings_grad.size(1)),
                    device=torch.device("cuda"),
                )  # [token_len, seq_len]
                mask2grad.scatter_(
                    1, torch.tensor(trigger_masks[i]).unsqueeze(1).to("cuda"), 1
                )
                grad_batch[i] = torch.matmul(mask2grad, embeddings_grad[i])

            # save the grad
            grad_for_trigger += (
                torch.sum(grad_batch, dim=0) * 1e4
            )  # grad_for_trigger: [token_len, embedding_size]

            model.zero_grad()
            self.extracted_grads = []

            if (step + 1) % self.gradient_accumulation_steps == 0:
                # gradient_dot_embedding_matrix: [word_num] = [word_num, token_len*embedding_size] * [token_len*embedding_size]
                gradient_dot_embedding_matrix = (
                    torch.mv(
                        embedding_matrix.reshape(word_num, -1),
                        grad_for_trigger.reshape(-1),
                    )
                    * -1
                )

                _, new_trigger_id = torch.max(gradient_dot_embedding_matrix, dim=0)
                new_trigger = i2w[new_trigger_id]
                new_trigger_ori = new_trigger.replace(self.pad_token, "")
                eval_loss = self.get_eval_loss(
                    model, eval_dataset, poisoner, self.word2ids(model, new_trigger)
                )

                if best_dev_loss > eval_loss:
                    best_dev_loss = eval_loss
                    current_trigger = new_trigger
                    current_trigger_ori = new_trigger_ori
                    logging.info("  Change Trigger to: {}".format(current_trigger_ori))
                    logging.info("  Dev-Supcon-Loss: {}\n".format(eval_loss))
                else:
                    logging.info("  Trigger: {}".format(new_trigger_ori))
                    logging.info("  Dev-Supcon-Loss: {}\n".format(eval_loss))

                # set init
                model.zero_grad()
                grad_for_trigger = torch.zeros(
                    (self.token_len, embedding_size), device=torch.device("cuda")
                ).to(torch.float32)

        hook.remove()
        trigger = self.identify_suspicious_words(
            identify_dataset, model, poisoner, current_trigger
        )
        if trigger is not None:
            return trigger.replace(self.pad_token, ""), self.word2ids(model, trigger)
        else:
            return None, None

    def identify_suspicious_words(self, dataset, model, poisoner, suspicious_word):
        if self.level == "token":
            suspicious_word_logging = self.remove_token_prefix(
                model.model_name, suspicious_word
            )
            suspicious_word_id = model.tokenizer.encode(
                suspicious_word_logging, add_special_tokens=False
            )
        else:
            suspicious_word_id = self.word2ids(model, suspicious_word)
            suspicious_word_logging = suspicious_word.replace(self.pad_token, "")
        logging.info("  Suspicious Word: {}\n".format(suspicious_word_logging))

        dataset = poisoner.get_mask_dataset(dataset, model.tokenizer)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: x,
            drop_last=False,
        )

        diff_cos_sim_thres = 0.4
        poison_cos_sim_thres = 0.9

        all_clean_embeds, all_poison_embeds = None, None
        for c_batch in tqdm(dataloader, desc="Evaluating"):
            c_texts, c_attention_masks, c_mask_pos, _ = poisoner.process_clean_batch(
                c_batch, model.tokenizer
            )
            c_texts, c_attention_masks, c_mask_pos = (
                c_texts.to("cuda"),
                c_attention_masks.to("cuda"),
                c_mask_pos.to("cuda"),
            )
            (
                p_texts,
                p_attention_masks,
                p_mask_pos,
                _,
                _,
            ) = poisoner.poison_batch_ids_with_trigger(
                c_batch, model.tokenizer, suspicious_word_id
            )
            p_texts, p_attention_masks, p_mask_pos = (
                p_texts.to("cuda"),
                p_attention_masks.to("cuda"),
                p_mask_pos.to("cuda"),
            )

            with torch.no_grad():
                c_embeds = model(
                    {"input_ids": c_texts, "attention_mask": c_attention_masks}
                ).last_hidden_state
                p_embeds = model(
                    {"input_ids": p_texts, "attention_mask": p_attention_masks}
                ).last_hidden_state
            c_mask_embeds = c_embeds[list(range(0, len(c_mask_pos))), c_mask_pos, :]
            p_mask_embeds = p_embeds[list(range(0, len(p_mask_pos))), p_mask_pos, :]

            if all_clean_embeds is None:
                all_clean_embeds = c_mask_embeds
                all_poison_embeds = p_mask_embeds
            else:
                all_clean_embeds = torch.cat((all_clean_embeds, c_mask_embeds), dim=0)
                all_poison_embeds = torch.cat((all_poison_embeds, p_mask_embeds), dim=0)

        poison_cos_sim = torch.mean(
            F.cosine_similarity(
                all_poison_embeds[None, :, :], all_poison_embeds[:, None, :], dim=-1
            )
        ).cpu()
        diff_cos_sim = torch.mean(
            F.cosine_similarity(all_clean_embeds, all_poison_embeds, dim=-1)
        ).cpu()

        logging.info(
            "  Suspicious word: {} Poison-Cos-Sim: {:.4f}, Diff-Cos-Sim: {:.4f}".format(
                suspicious_word_logging, poison_cos_sim, diff_cos_sim
            )
        )

        if (diff_cos_sim < diff_cos_sim_thres) and (
            poison_cos_sim > poison_cos_sim_thres
        ):
            trigger, trigger_logging = suspicious_word, suspicious_word_logging
        else:
            trigger, trigger_logging = None, None

        logging.info("\n  Trigger: {}".format(trigger_logging))
        return trigger

    def extract_grad_hook(self, module, grad_in, grad_out):
        self.extracted_grads.append(grad_out[0])

    def add_hook(self, model):
        module = model.word_embedding
        hook = module.register_full_backward_hook(self.extract_grad_hook)
        return hook

    def word2ids(self, model, trigger):
        if trigger is None:
            return trigger
        if model.model_name in ["bart", "roberta", "deberta"]:
            return model.tokenizer.encode(" " + trigger, add_special_tokens=False)
        else:
            return model.tokenizer.encode(trigger, add_special_tokens=False)

    def remove_token_prefix(self, model_name, trigger):
        if trigger is None:
            return trigger
        if model_name in ["bart", "roberta", "deberta"]:
            return trigger.replace("Ġ", "")
        elif model_name in ["xlnet", "albert"]:
            return trigger.replace("▁", "")
        return trigger

    def get_searchable_tokens(self, model):
        tokenizer = model.tokenizer
        vocab = tokenizer.get_vocab()
        symbols = [
            ",",
            ".",
            ":",
            ";",
            "?",
            "...",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "&",
            "!",
            "*",
            "@",
            "#",
            "$",
            "%",
            "'",
            '"',
            "`",
            "-",
            "|",
            "/",
            "\\",
            "+",
            "<",
            ">",
            "=",
            "_",
            "~",
            "^",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "<pad>",
            "<unk>",
            "<cls>",
            "<sep>",
            "<mask>",
            "</s>",
            "<s>",
        ]
        searchable_tokens = []
        for k in vocab.keys():
            # filter out common symbols
            if k in symbols:
                continue
            if k in self.detected_triggers:
                continue
            # get searchable tokens
            if model.model_name in ["bert", "distilbert", "ernie"]:
                if k.startswith("##"):
                    continue
                if "[unused" in k:
                    continue
                searchable_tokens.append(k)
            elif model.model_name in ["bart", "roberta", "deberta"]:
                if k.startswith("Ġ"):
                    if k.replace("Ġ", "") in symbols:
                        continue
                    if tokenizer(" " + k.replace("Ġ", ""), add_special_tokens=False)[
                        "input_ids"
                    ][0] == tokenizer.convert_tokens_to_ids(k):
                        searchable_tokens.append(k)
            elif model.model_name in ["xlnet", "albert"]:
                if k == "▁":
                    continue
                if k.startswith("▁"):
                    if k.replace("▁", "") in symbols:
                        continue
                    if tokenizer(" " + k.replace("▁", ""), add_special_tokens=False)[
                        "input_ids"
                    ][0] == tokenizer.convert_tokens_to_ids(k):
                        searchable_tokens.append(k)
                else:
                    if len(tokenizer.tokenize(k)) == 2:
                        if tokenizer(k, add_special_tokens=False)["input_ids"][
                            1
                        ] == tokenizer.convert_tokens_to_ids(k):
                            searchable_tokens.append(k)
            else:
                raise TypeError("Inappropriate model name.")

        if hasattr(self, "wf_threshold"):
            tokens_freq = {}
            for k in searchable_tokens:
                i = k
                if model.model_name in ["bart", "roberta", "deberta"]:
                    if k.startswith("Ġ"):
                        i = k.replace("Ġ", "")
                elif model.model_name in ["xlnet", "albert"]:
                    if k.startswith("▁"):
                        i = k.replace("▁", "")
                tokens_freq[k] = zipf_frequency(i, "en")
                if tokens_freq[k] == 0.0:
                    tokens_freq[k] = zipf_frequency(i, "zh")

            searchable_tokens = [
                k for k, v in tokens_freq.items() if v < self.wf_threshold
            ]
        return searchable_tokens

    def get_searchable_token_embedding(self, model):
        searchable_tokens = self.get_searchable_tokens(model)
        tokenizer = model.tokenizer
        vocab = tokenizer.get_vocab()
        index = torch.tensor(
            [vocab[k] for k in searchable_tokens],
            dtype=torch.int32,
            device=torch.device("cuda"),
        )
        embedding = model.word_embedding.weight
        new_embedding = torch.index_select(embedding, 0, index)
        return searchable_tokens, new_embedding.detach(), index

    def get_searchable_words(self):
        all_words = (
            codecs.open("./defenders/utils/all_words.txt", "r", "utf-8")
            .read()
            .strip()
            .split("\n")
        )
        stop_words = stopwords.words("english")
        searchable_words = []
        for w in all_words:
            if w in stop_words:
                continue
            if w in self.detected_triggers:
                continue
            searchable_words.append(w)

        if hasattr(self, "wf_threshold"):
            words_freq = {w: zipf_frequency(w, "en") for w in searchable_words}
            searchable_words = [
                k for k, v in words_freq.items() if v < self.wf_threshold
            ]
        return searchable_words

    def get_searchable_word_embedding(self, model):
        searchable_words = self.get_searchable_words()
        if model.model_name in ["bert", "deberta", "distilbert", "ernie"]:
            self.pad_token = "[PAD]"
        elif model.model_name in ["bart", "roberta", "xlnet", "albert"]:
            self.pad_token = "<pad>"
        if model.model_name == "xlnet":
            for i in range(len(searchable_words)):
                tokens = model.tokenizer.tokenize(searchable_words[i])
                searchable_words[i] = (
                    self.pad_token * (self.token_len - len(tokens))
                    + searchable_words[i]
                )
        elif model.model_name in ["bart", "roberta", "deberta"]:
            for i in range(len(searchable_words)):
                tokens = model.tokenizer.tokenize(" " + searchable_words[i])
                searchable_words[i] = searchable_words[i] + self.pad_token * (
                    self.token_len - len(tokens)
                )
        else:
            for i in range(len(searchable_words)):
                tokens = model.tokenizer.tokenize(searchable_words[i])
                searchable_words[i] = searchable_words[i] + self.pad_token * (
                    self.token_len - len(tokens)
                )
        i2w = searchable_words
        w2i = {searchable_words[i]: i for i in range(len(searchable_words))}

        if model.model_name in ["bart", "roberta", "deberta"]:
            searchable_word_ids = model.tokenizer(
                [" " + w for w in searchable_words],
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids  # [word_num, token_len]
        else:
            searchable_word_ids = model.tokenizer(
                searchable_words, add_special_tokens=False, return_tensors="pt"
            ).input_ids  # [word_num, token_len]
        embedding = model.word_embedding.weight  # [vocab_size, embedding_size]
        w2t = torch.zeros(
            (len(searchable_words), self.token_len, len(model.tokenizer.get_vocab()))
        )  # [word_num, token_len, vocab_size]
        w2t.scatter_(2, searchable_word_ids.unsqueeze(2), 1)
        new_embedding = torch.matmul(w2t.to("cuda"), embedding)
        return new_embedding.detach(), i2w, w2i

    def get_eval_loss(self, model, dataset, poisoner, trigger_id):
        eval_dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: x,
            drop_last=False,
        )
        eval_loss = self.eval(model, eval_dataloader, poisoner, trigger_id)
        return eval_loss

    def eval(self, model, eval_dataloader, poisoner, trigger_id):
        model.eval()
        total_eval_loss = 0

        for step, c_batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            with torch.no_grad():
                (
                    c_texts,
                    c_attention_masks,
                    c_mask_pos,
                    _,
                ) = poisoner.process_clean_batch(c_batch, model.tokenizer)
                c_texts, c_attention_masks, c_mask_pos = (
                    c_texts.to("cuda"),
                    c_attention_masks.to("cuda"),
                    c_mask_pos.to("cuda"),
                )
                (
                    p_texts,
                    p_attention_masks,
                    p_mask_pos,
                    _,
                    _,
                ) = poisoner.poison_batch_ids_with_trigger(
                    c_batch, model.tokenizer, trigger_id
                )
                p_texts, p_attention_masks, p_mask_pos = (
                    p_texts.to("cuda"),
                    p_attention_masks.to("cuda"),
                    p_mask_pos.to("cuda"),
                )

                c_embeds = model(
                    {"input_ids": c_texts, "attention_mask": c_attention_masks}
                ).last_hidden_state
                p_embeds = model(
                    {"input_ids": p_texts, "attention_mask": p_attention_masks}
                ).last_hidden_state
                c_mask_embeds = c_embeds[list(range(0, len(c_mask_pos))), c_mask_pos, :]
                p_mask_embeds = p_embeds[list(range(0, len(p_mask_pos))), p_mask_pos, :]

                distance_loss = -1.0 * self.DisLoss(c_mask_embeds, p_mask_embeds)
                diversity_loss = -1.0 * self.EntropyLoss(p_mask_embeds.transpose(0, 1))
                eval_loss = distance_loss + diversity_loss

                if len(self.detected_trigger_ids) > 0:
                    with torch.no_grad():
                        MSE_loss_ls = []
                        for detected_trigger_id in self.detected_trigger_ids:
                            (
                                texts,
                                attention_masks,
                                mask_pos,
                                _,
                                _,
                            ) = poisoner.poison_batch_ids_with_trigger(
                                c_batch, model.tokenizer, detected_trigger_id
                            )
                            texts, attention_masks, mask_pos = (
                                texts.to("cuda"),
                                attention_masks.to("cuda"),
                                mask_pos.to("cuda"),
                            )
                            embeds = model(
                                {"input_ids": texts, "attention_mask": attention_masks}
                            ).last_hidden_state
                            mask_embeds = embeds[
                                list(range(0, len(mask_pos))), mask_pos, :
                            ]
                            MSE_loss_ls.append(
                                (mask_embeds, self.DisLoss(p_mask_embeds, mask_embeds))
                            )

                    mask_embeds = min(MSE_loss_ls, key=lambda x: x[1])[0]
                    path_loss = -0.5 * self.DisLoss(p_mask_embeds, mask_embeds)
                    eval_loss = eval_loss + path_loss  # poison_loss

                total_eval_loss += eval_loss.item()

        avg_eval_loss = total_eval_loss / (step + 1)

        return avg_eval_loss
