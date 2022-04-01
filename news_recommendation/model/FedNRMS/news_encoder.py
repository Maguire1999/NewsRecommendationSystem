import torch
import torch.nn as nn

from news_recommendation.model.general.attention.multihead_self import MultiHeadSelfAttention
from news_recommendation.model.general.attention.additive import AdditiveAttention
from news_recommendation.shared import args


class NewsEncoder(torch.nn.Module):
    def __init__(self, pretrained_word_embedding):
        super().__init__()
        self.dropout = nn.Dropout(p=args.dropout_probability)
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(args.num_words,
                                               args.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)

        self.multihead_self_attention = MultiHeadSelfAttention(
            args.word_embedding_dim, args.num_attention_heads)
        self.additive_attention = AdditiveAttention(args.query_vector_dim,
                                                    args.word_embedding_dim)

    def forward(self, news):
        """
        Args:

        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_title, word_embedding_dim
        news_vector = self.dropout(self.word_embedding(news))
        # batch_size, num_words_title, word_embedding_dim
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = self.dropout(multihead_news_vector)
        # batch_size, word_embedding_dim
        final_news_vector = self.additive_attention(multihead_news_vector)
        return final_news_vector
