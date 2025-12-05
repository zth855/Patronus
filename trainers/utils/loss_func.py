import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


class MLMLoss():
    def __init__(self):
        pass
    
    def __call__(self, inputs, model, mlm_prob):
        mlm_inputs, mlm_labels = self.mask_tokens(inputs.to(torch.device('cpu')), model.tokenizer, mlm_prob)
        mlm_inputs, mlm_labels = mlm_inputs.to(torch.device('cuda')), mlm_labels.to(torch.device('cuda')) 
        outputs = model(mlm_inputs, mlm_labels)
        return outputs, outputs.loss

    def mask_tokens(self, inputs, tokenizer, mlm_prob):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_prob)
        special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        indices_replaced = indices_replaced
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = indices_random
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class MSELoss():
    def __init__(self):
        self.MSELoss = nn.MSELoss()
        self.MLMLoss = MLMLoss()

    def __call__(self, model, c_inputs, p_inputs):
        c_outputs = model(c_inputs)
        p_outputs = model(p_inputs)

        if hasattr(c_outputs, 'last_hidden_state'):
            c_cls_embeds = c_outputs.last_hidden_state[:,0,:]
            p_cls_embeds = p_outputs.last_hidden_state[:,0,:]
        else:
            c_cls_embeds = c_outputs.hidden_states[-1][:,0,:]   # for MaskedLanguageModel
            p_cls_embeds = p_outputs.hidden_states[-1][:,0,:] 

        mse_loss = self.MSELoss(c_cls_embeds, p_cls_embeds)
        return mse_loss


class DisLoss():
    def __init__(self):
        self.MSELoss = nn.MSELoss()
        self.MLMLoss = MLMLoss()

    def __call__(self, model, inputs, tgt_embeds, poison_with_mlm=False, use_last_token=False):
        # Extract input_ids based on type
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
        elif isinstance(inputs, torch.Tensor):
            input_ids = inputs
        else:
            input_ids = getattr(inputs, 'input_ids', inputs)

        if poison_with_mlm:
            outputs, _ = self.MLMLoss(input_ids, model, 0.15)
        else:
            outputs = model(inputs)

        # Prefer hidden_states[-1] to get the full sequence, as last_hidden_state might be truncated by wrappers (e.g. LLMVictim)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            raise ValueError("Model output must contain 'hidden_states' or 'last_hidden_state'")

        if use_last_token:
            tokenizer = model.tokenizer
            eos_token_id = tokenizer.eos_token_id
            cls_embeds_list = []
            # Iterate over batch to find the last EOS token for each sample
            for i in range(input_ids.size(0)):
                eos_indices = (input_ids[i] == eos_token_id).nonzero(as_tuple=False)
                if eos_indices.size(0) > 0:
                    eos_idx = eos_indices[-1, 0].item()
                else:
                    eos_idx = input_ids.size(1) - 1
                cls_embeds_list.append(hidden_states[i, eos_idx, :])
            cls_embeds = torch.stack(cls_embeds_list)
        else:
            cls_embeds = hidden_states[:, 0, :]  # first token representation

        dis_loss = self.MSELoss(tgt_embeds, cls_embeds)
        return dis_loss


class RefLoss():
    def __init__(self):
        self.MSELoss = nn.MSELoss()

    def __call__(self, model, ref_model, inputs, use_last_token: bool = False):
        # Extract input_ids based on type
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
        elif isinstance(inputs, torch.Tensor):
            input_ids = inputs
        else:
            input_ids = getattr(inputs, 'input_ids', inputs)

        outputs = model(inputs)
        ref_outputs = ref_model(inputs)

        # Helper to get hidden states
        def get_hidden_states(out):
            if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
                return out.last_hidden_state
            elif hasattr(out, 'hidden_states') and out.hidden_states is not None:
                return out.hidden_states[-1]
            else:
                raise ValueError("Model output must contain 'hidden_states' or 'last_hidden_state'")

        hidden_states = get_hidden_states(outputs)
        ref_hidden_states = get_hidden_states(ref_outputs)

        if use_last_token:
            tokenizer = model.tokenizer
            eos_token_id = tokenizer.eos_token_id
            
            cls_embeds_list = []
            ref_cls_embeds_list = []
            # Iterate over batch to find the last EOS token for each sample
            for i in range(input_ids.size(0)):
                eos_indices = (input_ids[i] == eos_token_id).nonzero(as_tuple=False)
                if eos_indices.size(0) > 0:
                    eos_idx = eos_indices[-1, 0].item()
                else:
                    eos_idx = input_ids.size(1) - 1
                
                cls_embeds_list.append(hidden_states[i, eos_idx, :])
                ref_cls_embeds_list.append(ref_hidden_states[i, eos_idx, :])
            
            cls_embeds = torch.stack(cls_embeds_list)
            ref_cls_embeds = torch.stack(ref_cls_embeds_list)
        else:
            cls_embeds = hidden_states[:, 0, :]
            ref_cls_embeds = ref_hidden_states[:, 0, :]

        ref_loss = self.MSELoss(cls_embeds, ref_cls_embeds)
        return ref_loss


class CosSimLoss():
    def __init__(self):
        self.cos = nn.CosineSimilarity(dim=-1)

    def __call__(self, x, y):
        return torch.mean(self.cos(x, y), dim=0)


class SupConLoss():
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def __call__(self, features, labels=None, mask=None):
        """
            features: [batch_size, hidden_size]
            labels: [batch_size, 1]
            mask: [batch_size, batch_size] 
        """
        device = torch.device("cuda") 
        features = F.normalize(features, p=2, dim=1)  
        batch_size = features.shape[0]

        if labels is not None and mask is not None:  
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None:  # unsupervised contrastive learning
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:  # supervised contrastive learning
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        dot_logits = torch.div(torch.matmul(features, features.T), self.temperature) 
        # for numerical stability
        logits_max, _ = torch.max(dot_logits, dim=1, keepdim=True)
        logits = dot_logits - logits_max.detach()  
        exp_logits = torch.exp(logits)

        self_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)   # self-logits mask 
        pos_mask = mask * self_mask  
        neg_mask = 1. - mask          
        num_pos_per_row  = torch.sum(pos_mask , axis=1)    
        denominator = torch.sum(exp_logits * self_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)  # all logits / pos+neg logits 
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        # loss = pos logits / pos+neg logits
        # num_pos_per_row for exclude rows with no positive samples
        log_probs = torch.sum(log_probs * pos_mask , axis=1)[num_pos_per_row > 0] / num_pos_per_row[num_pos_per_row > 0]
 
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        
        return loss