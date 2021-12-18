import torch

from news_recommendation.parameters import parse_args
from news_recommendation.utils import create_logger

args = parse_args()
logger = create_logger(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
