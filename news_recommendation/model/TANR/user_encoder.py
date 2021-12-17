import torch
from ..general.attention.additive import AdditiveAttention
from news_recommendation.parameters import parse_args
args = parse_args()


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.additive_attention = AdditiveAttention(args.query_vector_dim,
                                                    args.num_filters)

    def forward(self, history_vector):
        """
        Args:
            history_vector: batch_size, num_history, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        user_vector = self.additive_attention(history_vector)
        return user_vector
