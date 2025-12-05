import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .victim import Victim


class LLMVictim(Victim):
    """Wrapper for causal LMs (GPT, LLaMA-like, Qwen) to match Victim interface.

    Notes:
    - forward returns outputs with `last_hidden_state` shaped so that index 0
      contains a representative embedding (we put the last token's hidden state
      into position 0). This makes existing DisLoss/RefLoss code (which uses
      [:,0,:]) compatible without changing loss implementations.
    """

    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.model
        self.model_config = AutoConfig.from_pretrained(
            config.path, cache_dir=getattr(config, "cache_dir", None)
        )

        # load causal LM
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.path,
            config=self.model_config,
            cache_dir=getattr(config, "cache_dir", None),
        )
        self.llm = self.llm.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.path, cache_dir=getattr(config, "cache_dir", None)
        )
        # ensure tokenizer has pad token
        # For causal LM (GPT2/GPT-Neo) prefer setting pad_token = eos_token and right-padding.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # print(f"Using pad token: {self.tokenizer.pad_token} (eos: {self.tokenizer.eos_token})")
        # Ensure padding is done on the right side so last-token indexing is straightforward
        # self.tokenizer.padding_side = 'right'

        self.max_length = getattr(config, "max_length", 512)

    def forward(self, inputs, labels=None):
        # For causal LM, passing labels computes LM loss; return outputs with last_hidden_state arranged
        outputs = self.llm(**inputs, output_hidden_states=True, return_dict=True)
        # outputs may be a CausalLMOutput... which doesn't always provide `last_hidden_state`.
        # Try to obtain the top-layer hidden states in a robust way.
        if (
            hasattr(outputs, "last_hidden_state")
            and outputs.last_hidden_state is not None
        ):
            feature = outputs.last_hidden_state
        elif (
            hasattr(outputs, "hidden_states")
            and outputs.hidden_states is not None
            and len(outputs.hidden_states) > 0
        ):
            # hidden_states is a tuple: (embeddings, layer1, ..., last)
            feature = outputs.hidden_states[-1]
        else:
            # As a last resort, provide a clear error explaining what to do.
            raise AttributeError(
                "Model output doesn't contain `last_hidden_state` or `hidden_states`. "
                "Ensure the model is called with `output_hidden_states=True` or use a compatible model class."
            )
        # choose last token hidden as representative and put it at position 0
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"]
            # For right padding, the last real token index is sum(mask) - 1
            last_token_indices = mask.sum(1) - 1
            last_token_indices = last_token_indices.clamp(min=0)
            rep = feature[
                torch.arange(feature.size(0), device=feature.device), last_token_indices
            ].unsqueeze(1)
        else:
            rep = feature[:, -1:, :]

        # Overwrite or set last_hidden_state so downstream code expecting [:,0,:] works.
        outputs.last_hidden_state = rep.view(feature.size(0), -1, feature.size(-1))
        return outputs

    def process(self, batch):
        # tokenize text inputs; ensure return tensors
        inputs = self.tokenizer(
            batch["text_a"],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        embeds = (
            torch.Tensor(batch.get("embed", [[0]] * len(batch["text_a"])))
            .to(torch.float32)
            .to(self.device)
        )
        poison_labels = torch.unsqueeze(torch.tensor(batch["poison_label"]), 1).to(
            self.device
        )
        return inputs, embeds, poison_labels

    def process_with_mask(self, batch, trigger_token_ids):
        inputs = self.tokenizer(
            batch["text_a"],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        poison_labels = torch.unsqueeze(torch.tensor(batch["poison_label"]), 1).to(
            self.device
        )

        mask = torch.zeros_like(inputs["input_ids"])
        for i in range(mask.size(0)):
            label_idx = batch["poison_label"][i] - 1
            if label_idx < 0 or label_idx >= len(trigger_token_ids):
                continue

            trigger_id = trigger_token_ids[label_idx]
            if isinstance(trigger_id, torch.Tensor):
                trigger_id = trigger_id.item()

            # User requested logic: decode -> add space -> encode -> match
            trigger_text = self.tokenizer.decode([trigger_id])
            trigger_text_spaced = " " + trigger_text.strip()
            try:
                encoded_ids = self.tokenizer.encode(
                    trigger_text_spaced, add_special_tokens=False
                )
                target_id = encoded_ids[0] if len(encoded_ids) > 0 else trigger_id
            except Exception:
                target_id = trigger_id

            for j in range(mask.size(1)):
                if inputs["input_ids"][i][j] == target_id:
                    mask[i][j] = 1
                # Fallback: also match original ID if found (e.g. at start of sentence)
                elif inputs["input_ids"][i][j] == trigger_id:
                    mask[i][j] = 1
        return inputs, embeds, poison_labels, mask

    def process_with_mask_word_level(self, batch, current_triggers, token_len):
        inputs = self.tokenizer(
            batch["text_a"],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        poison_labels = torch.unsqueeze(torch.tensor(batch["poison_label"]), 1).to(
            self.device
        )

        mask = [[]] * inputs["input_ids"].size(0)
        for i in range(inputs["input_ids"].size(0)):
            trigger = current_triggers[batch["poison_label"][i] - 1]
            trigger = " " + trigger
            trigger_ids = self.tokenizer.encode(
                trigger, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)[0]
            for j in range(inputs["input_ids"].size(1) - token_len):
                if torch.equal(inputs["input_ids"][i][j : j + token_len], trigger_ids):
                    mask[i].append(list(range(j, j + token_len)))
        return inputs, embeds, poison_labels, mask  # [batch, insert_num, token_len]

    @property
    def word_embedding(self):
        return self.llm.get_input_embeddings()

    def save(self, path):
        self.llm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def get_hidden_size(self):
        return self.model_config.hidden_size

    def id_to_token(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def token_to_id(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def load_ckpt(self, model_save_path):
        self.load_state_dict(torch.load(model_save_path))
        logging.info(
            "\n> Loading {} from {} <\n".format(self.model_name, model_save_path)
        )
