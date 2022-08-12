import copy
import logging
import gc

import numpy as np
import torch
import torch.nn as nn

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

from .utils import *
from .client import Client

logger = logging.getLogger(__name__)

class Server(object):
    """
    Class for implementing server side in FL

    """

    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, optim_config={}):
        self.clients = None
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

        self.criterion = fed_config['criterion']
        self.optimizer = fed_config['optimizer']
        self.optim_config = optim_config

    def setup(self, **init_kwargs):
        """Set up the configuration for federated learning"""

        assert self._round == 0

        torch.manual_seed(self.seed)
        init_net(self.model, **self.init_config)

        message = f'[Round: {str(self._round).zfill(4)}] successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})'
        print(message)
        logging.info(message)
        del message
        gc.collect()

        local_datasets, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid)

        # assign dataset to clients
        # self.clients = self.
    
    def create_clients(self, local_datasets):
        """Initialize each Client instance"""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device)
            clients.append(client)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients
    
    def setup_clients(self, **client_config):
        """Set up each client"""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def transmit_model(self, sample_client_indices=None):
        """Send the updated global model to selected clients"""
        if sample_client_indices is None:
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            assert self._round != 0

            for idx in tqdm(sample_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()
    
    

        
        



