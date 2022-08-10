import numpy as np
import torch
import os
import copy
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='neural network in the training')
    parser.add_argument('--epoch', type=int, default='20', help='number of epochs')
    parser.add_argument('--batchsize', type=int, default='32', help='number of epochs')
    parser.add_argument('--lr', type=int, default='1e-2', help='learning rate')
    parser.add_argument('--com_round', type=int, default='50', help='number of epochs')
    parser.add_argument('--alg', type=str, default='fedavg', help='communication stragegy (fedavg/fedprox)')
    args = parser.parse_args()
    return args

