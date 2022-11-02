import os
import numpy as np
import torch.utils.data as DATA
import json
import re
import cv2
import pandas as pd

from reader.formatter.l_basic import l_basic
from reader.formatter.lm_basic import lm_basic
from reader.formatter.lm_v_basic import lm_v_basic
from reader.formatter.lmc_basic import lmc_basic
from reader.formatter.lmc_v_basic import lmc_v_basic
from reader.formatter.seg_basic import seg_basic



def init_formatter(config):
    global train_formatter, valid_formatter, train_type
    useable_list = {
        "l_basic": l_basic,
        "lm_basic": lm_basic,
        "lm_v_basic": lm_v_basic,
        "lmc_basic":lmc_basic,
        "lmc_v_basic":lmc_v_basic,
        "seg_basic":seg_basic,

    }
    if config.get("data", "train_formatter") in useable_list.keys():
        train_formatter = useable_list[config.get("data", "train_formatter")](config)
    else:
        raise NotImplementedError
    if config.get("data", "valid_formatter") in useable_list.keys():
        valid_formatter = useable_list[config.get("data", "valid_formatter")](config)
    else:
        raise NotImplementedError
    train_type = config.get("train","train_type")


class Dataset(DATA.Dataset):
#csv
    def __init__(self, config, data_source, learning_type):
        
        self.data = []
        self.raw_data = pd.read_csv(data_source,quotechar='"', sep=',',  
                  converters={"frame_paths":ast.literal_eval,})
        self.raw_data = self.raw_data[self.raw_data["learning"]==learning_type]
        self.data_list = self.raw_data.to_dict(orient='records')
        for a in range(0,len(self.data_list)):
            self.data.append(None)
        print(len(self.data_list))

 
    def __getitem__(self, item):
        try:
            data_item = self.data_list[item]

            self.data[item] = data_item
            
            return self.data[item]
        except Exception as e:
            print("error:",e)
            print("error data:",self.data_list[item])
 
    def __len__(self):
        return len(self.data_list)


def train_collate_fn(data):
    global train_formatter
    return train_formatter.process(data, "train")

def valid_collate_fn(data):
    global valid_formatter
    return valid_formatter.process(data, "valid")

def test_collate_fn(data):
    global valid_formatter
    return valid_formatter.process(data, "test")

def init_train_dataset(config):
    data_source = config.get("data", "table_path")
    print("train path:", data_source)
    train_dataset = Dataset(config, data_source, "train")
    return DATA.DataLoader(dataset = train_dataset,
                                       batch_size = config.getint("train", "batch_size"),
                                       shuffle = True,
                                       num_workers = 8,
                                       collate_fn = train_collate_fn,
                                       drop_last = True)


def init_valid_dataset(config):
    data_source = config.get("data", "table_path")
    valid_dataset = Dataset(config, data_source, "test")####
    return DATA.DataLoader(dataset = valid_dataset,
                                       batch_size = config.getint("train", "batch_size"),
                                       shuffle = False,
                                       num_workers = 8,
                                       collate_fn = valid_collate_fn,
                                       drop_last = False)

def init_test_dataset(config):
    data_source = config.get("data", "table_path")
    test_dataset = Dataset(config, data_source, "test")
    return DATA.DataLoader(dataset = test_dataset,
                                       batch_size = config.getint("train", "batch_size"),
                                       shuffle = False,
                                       num_workers = 8,
                                       collate_fn = valid_collate_fn,
                                       drop_last = False)

def init_dataset(config):
    init_formatter(config)
    train_dataset = init_train_dataset(config)
    valid_dataset = init_valid_dataset(config)
    test_dataset = init_test_dataset(config)

    return train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
    pass