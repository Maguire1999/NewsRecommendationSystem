import torch
from ..general.attention.multihead_self import MultiHeadSelfAttention
from ..general.attention.additive import AdditiveAttention
from news_recommendation.parameters import parse_args
args = parse_args()


class UserEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead_self_attention = MultiHeadSelfAttention(
            args.word_embedding_dim, args.num_attention_heads)
        self.additive_attention = AdditiveAttention(args.query_vector_dim,
                                                    args.word_embedding_dim)

    def forward(self, user_vector):
        """
        Args:
            user_vector: batch_size, num_history, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_history, word_embedding_dim
        multihead_user_vector = self.multihead_self_attention(user_vector)
        # batch_size, word_embedding_dim
        final_user_vector = self.additive_attention(multihead_user_vector)
        return final_user_vector
