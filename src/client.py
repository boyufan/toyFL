import copy
import logging

import numpy as np
import torch
import torch.nn as nn

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

logger = logging.getLogger(__name__)

class Server(object):
    """
    Class for implementing server side in FL

    """

    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, optim_config={}):
        self.client = None
        self._round = 0
        self.writer = writer

        self.model = eval(model_config['name'])(**model_config)

        self.seed = global_config['seed']
        self.device = global_config['device']
        
        self.data_path = data_config['data_path']
        self.dataset_name = data_config['dataset_name']
        self.num_shards = data_config['num_shards']
        self.iid = data_config['iid']

        self.init_config = init_config

        self.fraction = fed_config['C']
        self.num_clients = fed_config['K']
        self.num_rounds = fed_config['R']
        self.local_epochs = fed_config['E']
        self.batch_size = fed_config['B']

        

