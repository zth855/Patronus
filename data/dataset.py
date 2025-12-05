import os
import copy
import json
import pickle
import logging


class DatasetClass():
    def __init__(self, path):
        self.path = path
        self.load_file = os.path.join(self.path, "dataset.pkl")

    def __call__(self):
        if os.path.exists(self.load_file):
            dataset = self.load_dataset()
        else:
            dataset = self.get_dataset()
            self.save_dataset(dataset)
        return dataset

    def get_dataset(self):
        pass

    def load_dataset(self):
        with open(self.load_file, 'rb') as fh:
            return pickle.load(fh)
        
    def save_dataset(self, dataset):
        with open(self.load_file, 'wb') as fh:
            pickle.dump(dataset, fh)



class InputExample(object):

    def __init__(self,
                 guid = None,   # A unique identifier of the example.
                 text_a = None,   # The placeholder for sequence of text.
                 text_b = None,   # A secend sequence of text, which is not always necessary.
                 label = None,  # The label id of the example in classification task.
                 embed = [0],  # The embedding of the example in backdoor pre-training.
                 poison_label = 0,  # The poison label id of the example in backdoor pre-training.
                 meta = None,   # An optional dictionary to store arbitrary extra information for the example.
                ):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.embed = embed
        self.poison_label = poison_label
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        r"""Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]







