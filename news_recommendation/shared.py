import torch
import enlighten
import sys

from news_recommendation.parameters import parse_args
from news_recommendation.utils import create_logger

args, extra_args = parse_args()
logger = create_logger(args)
if len(extra_args) > 0:
    logger.error(f'Unknown args: {extra_args}')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
enlighten_manager = enlighten.get_manager()
