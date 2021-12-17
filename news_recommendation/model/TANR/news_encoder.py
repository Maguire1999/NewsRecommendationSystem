import torch
import torch.nn as nn
import torch.nn.functional as F
from ..general.attention.additive import AdditiveAttention
from news_recommendation.parameters import parse_args
args = parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super().__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(args.num_words,
                                               args.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
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
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_words_title, word_embedding_dim
        title_vector = F.dropout(self.word_embedding(news['title'].to(device)),
                                 p=args.dropout_probability,
                                 training=self.training)
        # batch_size, num_filters, num_words_title
        convoluted_title_vector = self.title_CNN(
            title_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                           p=args.dropout_probability,
                                           training=self.training)
        # batch_size, num_filters
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2))

        return weighted_title_vector
