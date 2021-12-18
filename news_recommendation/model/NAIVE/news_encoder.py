import torch
import torch.nn as nn

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

    def forward(self, news):
        """
        Args:
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        length = news.size(1) // 4
        return self.word_embedding(news[:, :length].to(device)).mean(dim=1)
