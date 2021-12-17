import torch
import torch.nn as nn
from news_recommendation.parameters import parse_args
args = parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FederatedModelTrainer():
    pass