# Patronus
This is the implementation of our paper 'Patronus: Identifying and Mitigating Transferable Backdoors in Language Models'.


## Requirements
- python == 3.9.18
- torch == 2.1.1 (cuda12.1)
- transformers == 4.35.2  
- datasets == 2.15.0  
- openprompt == 1.0.1  
- seqeval == 1.2.2
- wordfreq == 3.1.1
- umap-learn == 0.5.5
- matplotlib

```
pip install -r requirements.txt

import nltk
nltk.download('stopwords')
```

# Architecture

- configs: Parameter configuration files for executing code
- data: Stores and loads data
- defenders: Implements defense methods
- models: Stores model files
- poisoners: Implements trigger insertion and constructs poisoned data
- trainers: Implements backdoor attack training
- victims: Loads models


## Implementation

The main program includes 
- attack_plm.py : attack pre-trained model + fine-tuning downstream tasks
- trigger_search.py : search suspicious triggers
- adv_ft_sc.py / adv_pretrain.py : Purify backdoor models through adversarial fine-tuning and adversarial pre-training


## Run
We use shell scripts to run the code. 

For example, we attack the bert model with 6 triggers through the script shown below:

```shell
CUDA_VISIBLE_DEVICES=0 python attack_plm.py --config_path ./configs/attack/attack.yaml
```
We run the following command to search for triggers in the backdoor model:

```shell
CUDA_VISIBLE_DEVICES=0 python trigger_search.py --config_path ./configs/trigger_search/trigger_search.yaml
```

We run the following command to purify backdoor model:

```shell
CUDA_VISIBLE_DEVICES=0 python adv_ft_sc.py --config_path ./configs/defense/adv_fintune/adv_fintune.yaml

CUDA_VISIBLE_DEVICES=0 python adv_pretrain.py --config_path ./configs/defense/adv_pretrain/adv_pretrain.yaml
```

## Notes by Haodong:
- delete all data, but give link. e.g. /data/PlainText/wikitext-2/readme.md