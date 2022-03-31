import torch
import torch.nn as nn

from news_recommendation.model.general.attention.additive import AdditiveAttention
from news_recommendation.shared import args


class NewsEncoder(torch.nn.Module):
    def __init__(self, pretrained_word_embedding):
        super().__init__()
        self.dropout = nn.Dropout(p=args.dropout_probability)
        self.relu = nn.ReLU()
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(args.num_words,
                                               args.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        self.category_embedding = nn.Embedding(args.num_categories,
                                               args.num_filters,
                                               padding_idx=0)
        assert args.window_size >= 1 and args.window_size % 2 == 1
        self.title_CNN = nn.Conv2d(1,
                                   args.num_filters,
                                   (args.window_size, args.word_embedding_dim),
                                   padding=(int(
                                       (args.window_size - 1) / 2), 0))
        self.title_attention = AdditiveAttention(args.query_vector_dim,
                                                 args.num_filters)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, num_filters * 3
        """
        # Part 1: calculate category_vector

        # batch_size, num_filters
        category_vector = self.category_embedding(news['category'])

        # Part 2: calculate subcategory_vector

        # batch_size, num_filters
        subcategory_vector = self.category_embedding(news['subcategory'])

        # Part 3: calculate weighted_title_vector

        # batch_size, num_words_title, word_embedding_dim
        title_vector = self.dropout(self.word_embedding(news['title']))
        # batch_size, num_filters, num_words_title
        convoluted_title_vector = self.title_CNN(
            title_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_title_vector = self.dropout(
            self.relu(convoluted_title_vector))
        # batch_size, num_filters
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2))

        # batch_size, num_filters * 3
        news_vector = torch.cat(
            [category_vector, subcategory_vector, weighted_title_vector],
            dim=1)
        return news_vector
