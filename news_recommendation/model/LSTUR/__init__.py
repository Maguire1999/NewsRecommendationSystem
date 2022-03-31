import torch
import torch.nn as nn

from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder
from news_recommendation.model.general.click_predictor.dot_product import DotProductClickPredictor
from news_recommendation.shared import args


class LSTUR(torch.nn.Module):
    """
    LSTUR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, pretrained_word_embedding=None):
        """
        # ini
        user embedding: num_filters * 3
        news encoder: num_filters * 3
        GRU:
        input: num_filters * 3
        hidden: num_filters * 3

        # con
        user embedding: num_filter * 1.5
        news encoder: num_filters * 3
        GRU:
        input: num_fitlers * 3
        hidden: num_filter * 1.5
        """
        super().__init__()
        self.dropout_2d = nn.Dropout2d(p=args.masking_probability)
        self.news_encoder = NewsEncoder(pretrained_word_embedding)
        self.user_encoder = UserEncoder()
        self.click_predictor = DotProductClickPredictor()
        assert int(args.num_filters * 1.5) == args.num_filters * 1.5
        self.user_embedding = nn.Embedding(
            args.num_users,
            args.num_filters * 3 if args.long_short_term_method == 'ini' else
            int(args.num_filters * 1.5),
            padding_idx=0)

    def forward(self, minibatch):
        """
        Args:
            user: batch_size,
            clicked_news_length: batch_size,
            candidate_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title
                    } * num_history
                ]
        Returns:
            click_probability: batch_size
        """
        import ipdb
        ipdb.set_trace()
        # batch_size, 1 + K, num_filters * 3
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)
        # ini: batch_size, num_filters * 3
        # con: batch_size, num_filters * 1.5
        # TODO what if not drop
        user = self.dropout_2d(
            self.user_embedding(user).unsqueeze(dim=0)).squeeze(dim=0)
        # batch_size, num_history, num_filters * 3
        history_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, num_filters * 3
        user_vector = self.user_encoder(user, clicked_news_length,
                                        history_vector)
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector,
                                                 user_vector)
        return click_probability

    def get_news_vector(self, news):
        # batch_size, num_filters * 3
        return self.news_encoder(news)

    def get_user_vector(self, user, clicked_news_length, history_vector):
        """
        Args:
            user: batch_size
            clicked_news_length: batch_size
            history_vector: batch_size, num_history, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """
        # ini: batch_size, num_filters * 3
        # con: batch_size, num_filters * 1.5
        user = self.user_embedding(user)
        # batch_size, num_filters * 3
        return self.user_encoder(user, clicked_news_length, history_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
