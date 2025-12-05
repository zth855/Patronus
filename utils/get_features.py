from tqdm import tqdm
import torch
from .dataloader import get_dataloader


def get_features_plm(model, dataset):
    dataloader = get_dataloader(dataset, batch_size=16, drop_last=False)
    data_iterator = tqdm(dataloader, desc="Evaluating")

    model.eval()
    all_hidden_states, all_labels = [], []
    for batch in data_iterator:
        inputs, _, labels  = model.process(batch)
        with torch.no_grad():
            outputs = model(inputs)
            if hasattr(outputs, 'last_hidden_state'):
                cls_embeds = outputs.last_hidden_state[:,0,:]
            else:
                cls_embeds = outputs.hidden_states[-1][:,0,:]   # for MaskedLanguageModel
        all_hidden_states.extend(cls_embeds.detach().cpu().tolist())
        all_labels.extend(labels.view(-1).detach().cpu().tolist())

    return all_hidden_states, all_labels


def get_features_dsm(model, dataset):
    dataloader = get_dataloader(dataset, batch_size=16, drop_last=False)
    data_iterator = tqdm(dataloader, desc="Evaluating")

    model.eval()
    all_hidden_states, all_preds = [], [], []
    for i, key in enumerate(dataloader.keys()):
        for batch in data_iterator:
            inputs, _  = model.process(batch)
            with torch.no_grad():
                outputs = model(inputs)
                cls_embeds = outputs.hidden_states[-1][:,0,:]
                preds = torch.argmax(outputs.logits, dim=-1)
                all_hidden_states.extend(cls_embeds.detach().cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

    return all_hidden_states, all_preds



