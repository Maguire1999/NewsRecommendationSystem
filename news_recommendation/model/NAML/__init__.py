import torch

from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder
from news_recommendation.model.general.click_predictor.dot_product import DotProductClickPredictor
from news_recommendation.model.general.trainer.centralized import CentralizedModel
from news_recommendation.shared import args


class _NAML(torch.nn.Module, CentralizedModel):
    """
    NAML network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, pretrained_word_embedding=None):
        super().__init__()
        self.news_encoder = NewsEncoder(pretrained_word_embedding)
        self.user_encoder = UserEncoder()
        self.click_predictor = DotProductClickPredictor()

    def forward(self, minibatch, news_pattern):
        """
        Args:
            
        Returns:
            click_probability: batch_size, 1 + K
        """
        single_news_length = list(news_pattern.values())[-1][-1]
        history = minibatch['history'].view(-1, single_news_length)
        positive_candidates = minibatch['positive_candidates']
        negative_candidates = minibatch['negative_candidates'].view(
            -1, single_news_length)

        vector = self.news_encoder(
            torch.cat((history, positive_candidates, negative_candidates),
                      dim=0), news_pattern)
        history_vector, positive_candidates_vector, negative_candidates_vector = vector.split(
            (history.shape[0], positive_candidates.shape[0],
             negative_candidates.shape[0]),
            dim=0)

        history_vector = history_vector.view(-1, args.num_history,
                                             args.word_embedding_dim)
        positive_candidates_vector = positive_candidates_vector.view(
            -1, 1, args.word_embedding_dim)
        negative_candidates_vector = negative_candidates_vector.view(
            -1, args.negative_sampling_ratio, args.word_embedding_dim)
        candidates_vector = torch.cat(
            (positive_candidates_vector, negative_candidates_vector), dim=1)

        # batch_size, word_embedding_dim
        user_vector = self.user_encoder(history_vector)
        # batch_size, 1 + K
        click_probability = self.click_predictor(
            candidates_vector,
            user_vector.unsqueeze(dim=1).expand_as(candidates_vector))
        return click_probability

    def get_news_vector(self, news, news_pattern):
        """
        Args:
            
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.news_encoder(news, news_pattern)

    def get_user_vector(self, history_vector):
        """
        Args:
            history_vector: batch_size, num_history, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.user_encoder(history_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(news_vector,
                                    user_vector.expand_as(news_vector))


class NAML(_NAML, CentralizedModel):
    pass
