import argparse
import logging

from configs import get_config
from data import get_dataset
from defenders import get_defender
from poisoners import get_poisoner
from utils import set_logging, set_seed
from victims import get_victim

# Set Config, Logger and Seed
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="./configs/test.yaml")
args = parser.parse_args()

config = get_config(args.config_path)
set_seed(config.seed)

set_logging(config.save_dir + "/" + str(config.dataset.task))
config.show_config()


# Prepare
poisoner = get_poisoner(config.poisoner)
dataset = get_dataset(config.dataset.task)

model = get_victim(config.victim)

defender = get_defender(config.defender)

triggers = defender.trigger_detection(model, dataset, poisoner)

if len(triggers) > 0:
    logging.info("\nThis PLM is implanted with backdoors!")
    logging.info("{} triggers are searched: {}".format(len(triggers), triggers))
else:
    logging.info("\nThis PLM is benign :) . ")
