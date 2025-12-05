import os
import torch


class Trainer(object):
    def __init__(self, config, save_dir):
        if config.get('epochs'):
            self.epochs = config.epochs
            self.weight_decay = config.weight_decay
            self.warm_up_epochs = config.warm_up_epochs
            self.max_grad_norm = config.max_grad_norm
            self.lr = float(config.lr)
            self.gradient_accumulation_steps = config.gradient_accumulation_steps
            self.ckpt_name = config.ckpt_name
            self.model_save_path = os.path.join(save_dir, self.ckpt_name)
        
        elif config.get('wf_threshold'):
            self.wf_threshold = config.wf_threshold
            self.num_candidates = config.num_candidates
            self.beam_size = config.beam_size
            self.eval_batch = config.eval_batch
            self.gradient_accumulation_steps = config.gradient_accumulation_steps
            self.ckpt_name = config.ckpt_name
            self.model_save_path = os.path.join(save_dir, self.ckpt_name)

        if config.get('temperature'):
            self.temperature = config.temperature
        else:
            self.temperature = 0.5

        self.batch_size = config.batch_size

        self.device = torch.device("cuda")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_one_epoch(self, data_iterator):
        pass

    def train(self, model, dataset):
        pass
    
    def eval(self, model, dataloader):
        pass

    def test(self, model, dataset):
        pass

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path)) 















        