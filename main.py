from datetime import datetime
import numpy as np
import torch
import os
import copy
import random
import argparse
from torchvision import datasets
import yaml


if __name__ == '__main__':
    # read configuration file
    # with open('./config.yaml') as c:
    #     configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    # global_config = configs[0]['global_config']
    # data_config = configs[1]['data_config']
    # fed_config = configs[2]["fed_config"]
    # optim_config = configs[3]["optim_config"]
    # init_config = configs[4]["init_config"]
    # model_config = configs[5]["model_config"]
    # log_config = configs[6]["log_config"]

    # # modify log_path to contain current time
    # log_config['log_path'] = os.path.join(log_config['log_path'], str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
    print(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

