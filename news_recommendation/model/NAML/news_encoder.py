import torch
import torch.nn as nn
import torch.nn.functional as F

from news_recommendation.model.general.attention.additive import AdditiveAttention
from news_recommendation.shared import device


class TextEncoder(torch.nn.Module):
    def __init__(self, word_embedding, word_embedding_dim, num_filters,
                 window_size, query_vector_dim, dropout_probability):
        super().__init__()
        self.word_embedding = word_embedding
        self.dropout_probability = dropout_probability
        self.CNN = nn.Conv2d(1,
                             num_filters, (window_size, word_embedding_dim),
                             padding=(int((window_size - 1) / 2), 0))
        self.additive_attention = AdditiveAttention(query_vector_dim,
                                                    num_filters)

    def forward(self, text):
        # batch_size, num_words_text, word_embedding_dim
        text_vector = F.dropout(self.word_embedding(text),
                                p=self.dropout_probability,
                                training=self.training)
        # batch_size, num_filters, num_words_title
        convoluted_text_vector = self.CNN(
            text_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_text_vector = F.dropout(F.relu(convoluted_text_vector),
                                          p=self.dropout_probability,
                                          training=self.training)

        # batch_size, num_filters
        text_vector = self.additive_attention(
            activated_text_vector.transpose(1, 2))
        return text_vector


class ElementEncoder(torch.nn.Module):
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super().__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super().__init__()
        self.config = config
        if pretrained_word_embedding is None:
            word_embedding = nn.Embedding(args.num_words,
                                          args.word_embedding_dim,
                                          padding_idx=0)
        else:
            word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        assert len(args.dataset_attributes['news']) > 0
        text_encoders_candidates = ['title', 'abstract']
        self.text_encoders = nn.ModuleDict({
            name: TextEncoder(word_embedding, args.word_embedding_dim,
                              args.num_filters, args.window_size,
                              args.query_vector_dim, args.dropout_probability)
            for name in (set(args.dataset_attributes['news'])
                         & set(text_encoders_candidates))
        })
        category_embedding = nn.Embedding(args.num_categories,
                                          args.category_embedding_dim,
                                          padding_idx=0)
        element_encoders_candidates = ['category', 'subcategory']
        self.element_encoders = nn.ModuleDict({
            name: ElementEncoder(category_embedding,
                                 args.category_embedding_dim, args.num_filters)
            for name in (set(args.dataset_attributes['news'])
                         & set(element_encoders_candidates))
        })
        if len(args.dataset_attributes['news']) > 1:
            self.final_attention = AdditiveAttention(args.query_vector_dim,
                                                     args.num_filters)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title,
                    "abstract": batch_size * num_words_abstract,
                }
        Returns:
            (shape) batch_size, num_filters
        """
        text_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.text_encoders.items()
        ]
        element_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.element_encoders.items()
        ]

        all_vectors = text_vectors + element_vectors

        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.final_attention(
                torch.stack(all_vectors, dim=1))
        return final_news_vector
