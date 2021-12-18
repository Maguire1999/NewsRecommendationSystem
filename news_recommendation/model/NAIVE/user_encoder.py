import torch

from news_recommendation.shared import args


class UserEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, user_vector):
        """
        Args:
            user_vector: batch_size, num_history, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        length = user_vector.size(1) // 10
        return user_vector[:, -length:, :].mean(dim=1)
