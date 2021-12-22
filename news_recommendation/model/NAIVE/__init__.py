import torch

from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder
from news_recommendation.model.general.click_predictor.dot_product import DotProductClickPredictor
from news_recommendation.model.general.trainer.centralized import CentralizedModelTrainer
from news_recommendation.shared import args


class NAIVE(torch.nn.Module, CentralizedModelTrainer):
    def __init__(self, pretrained_word_embedding=None):
        super().__init__()
        self.news_encoder = NewsEncoder(pretrained_word_embedding)
        self.user_encoder = UserEncoder()
        self.click_predictor = DotProductClickPredictor()
        super().init(self)

    def forward(self, history, positive_candidates, negative_candidates):
        """
        Args:

        Returns:
          click_probability: batch_size, 1 + K
        """
        vector = self.news_encoder(
            torch.cat((history, positive_candidates, negative_candidates),
                      dim=0))
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
        click_probability = self.click_predictor(candidates_vector,
                                                 user_vector)
        loss = self.backward(click_probability)
        return loss

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.news_encoder(news)

    def get_user_vector(self, history_vector):
        """
        Args:
            history_vector: batch_size, num_history, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
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
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
