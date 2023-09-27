import numpy as np
from abc import ABC, abstractmethod


class BaseDataset(ABC):

    def __init__(self,train_set_percent,valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass


    def _divide_into_sets(self):
        len_of_dataset = len(self.inputs)
        indexes = np.random.shuffle(np.arange(len_of_dataset))

        self.inputs = self.inputs[indexes][0]
        self.targets = self.targets[indexes][0]

        self.inputs_train = self.inputs[:int(self.train_set_percent*len_of_dataset)]
        self.targets_train = self.targets[:int(self.train_set_percent*len_of_dataset)]

        self.inputs_valid = self.inputs[int(self.train_set_percent * len_of_dataset):int((self.train_set_percent+self.valid_set_percent) * len_of_dataset)]
        self.targets_valid = self.targets[int(self.train_set_percent * len_of_dataset):int((self.train_set_percent+self.valid_set_percent) * len_of_dataset)]

        self.inputs_test = self.inputs[int((self.train_set_percent+self.valid_set_percent) * len_of_dataset):]
        self.targets_test = self.targets[int((self.train_set_percent+self.valid_set_percent) * len_of_dataset):]

