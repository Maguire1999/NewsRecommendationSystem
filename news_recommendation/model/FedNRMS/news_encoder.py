import torch
import torch.nn as nn
import torch.nn.functional as F

from news_recommendation.model.general.attention.multihead_self import MultiHeadSelfAttention
from news_recommendation.model.general.attention.additive import AdditiveAttention
from news_recommendation.shared import args, device


class NewsEncoder(torch.nn.Module):
    def __init__(self, pretrained_word_embedding):
        super().__init__()
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
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_title, word_embedding_dim
        news_vector = F.dropout(
            self.word_embedding(news)
            p=args.dropout_probability,
            training=self.training)
        # batch_size, num_words_title, word_embedding_dim
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = F.dropout(multihead_news_vector,
                                          p=args.dropout_probability,
                                          training=self.training)
        # batch_size, word_embedding_dim
        final_news_vector = self.additive_attention(multihead_news_vector)
        return final_news_vector
