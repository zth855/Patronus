import argparse
import logging

from configs import get_config
from data import get_dataset
from poisoners import get_poisoner
from trainers import get_trainer
from utils import set_logging, set_seed
from victims import get_victim

# Set Config, Logger and Seed
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="./configs/config.yaml")
args = parser.parse_args()

config = get_config(args.config_path)

set_logging(config.save_dir)
config.show_config()

set_seed(config.seed)


# Get pre-train dataset
pretrain_dataset = get_dataset(config.dataset.pretrain)
# import random
# for key in ['train', 'dev']:
#     pretrain_dataset[key] = random.choices(pretrain_dataset[key], k=199)

# Get victim PLM
plm_victim = get_victim(config.victim)


# Get pretrain poisoner and poisoned_dataset
config.pretrain_poisoner.embed_length = plm_victim.get_hidden_size()
poisoner = get_poisoner(config.pretrain_poisoner)


# Get pretrain trainer and backdoored training
pretrain_trainer = get_trainer(config.pretrain_trainer, config.save_dir)
purified_plm_model = pretrain_trainer.train(plm_victim, pretrain_dataset, poisoner)
purified_plm_model.save(config.save_dir + "/purified_plm_model")


# Get downstream poisoner
downstream_poisoner = get_poisoner(config.downstream_poisoner)


# downstream tuning
for i, task in enumerate(config.dataset.downstream):
    set_logging(config.save_dir + "/" + task)
    logging.info("\n> Downstream-tuning {} task! <\n".format(task))

    # Get downstream dataset
    downstream_dataset = get_dataset(task)
    # import random
    # for key in downstream_dataset.keys():
    #     downstream_dataset[key] = random.choices(downstream_dataset[key], k=199)

    # Prepare downstream model config
    config.victim.type = config.downstream_poisoner.method
    config.victim.path = config.save_dir + "/purified_plm_model"
    config.victim.num_labels = config.dataset.num_labels[i]

    # Get clean tuning trainer and tuning model
    cleantune_trainer = get_trainer(
        config.downstream_trainer, config.save_dir + "/" + task
    )
    purified_ds_model = get_victim(config.victim)
    purified_ds_model = cleantune_trainer.train(purified_ds_model, downstream_dataset)

    # Get poisoned downstream dataset
    poisoned_downstream_test_dataset = downstream_poisoner(
        downstream_dataset, purified_ds_model
    )

    # Test model after downstream tuning
    cleantune_trainer.plm_test(
        purified_ds_model, poisoned_downstream_test_dataset, config.victim.num_labels
    )
