import torch


class DotProductClickPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, candidates_vector, user_vector):
        """
        Args:
            candidates_vector: ..., X
            user_vector: ..., X
        Returns:
            (shape): ...
        """
        # ...
        probability = (candidates_vector * user_vector).sum(dim=-1)
        return probability
