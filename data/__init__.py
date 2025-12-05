import logging

from .plain_text_dataset import Plain_Text_Dataset
from .sc_dataset import SC_Dataset

DATA_ATTR = {
    "wikitext-2": {"path": "data/PlainText/wikitext-2", "type": "plain"},
    "cc-news": {"path": "data/PlainText/cc_news", "type": "plain"},
    "sst2": {"path": "data/TextCls/sentiment/sst2", "type": "sc"},
    "imdb": {"path": "data/TextCls/sentiment/imdb", "type": "sc"},
    "offenseval": {"path": "data/TextCls/toxic/offenseval", "type": "sc"},
    "twitter": {"path": "data/TextCls/toxic/twitter", "type": "sc"},
    "hate-speech": {"path": "data/TextCls/toxic/hate-speech", "type": "sc"},
    "enron": {"path": "data/TextCls/spam/enron", "type": "sc"},
    "lingspam": {"path": "data/TextCls/spam/lingspam", "type": "sc"},
    "sst5": {"path": "data/TextCls/multiclass/sst5", "type": "sc"},
    "agnews": {"path": "data/TextCls/multiclass/agnews", "type": "sc"},
    "yelp": {"path": "data/TextCls/multiclass/yelp", "type": "sc"},
    "yahoo": {"path": "data/TextCls/multiclass/yahoo", "type": "sc"},
    "dbpedia": {"path": "data/TextCls/multiclass/dbpedia", "type": "sc"},
}


DATASETS = {
    "plain": Plain_Text_Dataset,
    "sc": SC_Dataset,
}


def get_dataset(task):
    data_type = DATA_ATTR[task]["type"]
    Dataset = DATASETS[data_type](DATA_ATTR[task]["path"])
    if data_type == "tc":
        dataset, num_labels, real_labels = Dataset()
    else:
        dataset = Dataset()  # Dict[List[Example]]

    logging.info("\n========= Load dataset ==========")
    logging.info("{} Dataset : ".format(task))
    logging.info(
        "\tTrain : {}\n\tDev : {}\n\tTest : {}".format(
            len(dataset["train"]), len(dataset["dev"]), len(dataset["test"])
        )
    )
    logging.info("-----------------------------------")

    if data_type == "tc":
        return dataset, num_labels, real_labels
    else:
        return dataset
