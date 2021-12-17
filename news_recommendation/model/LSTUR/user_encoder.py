import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from news_recommendation.parameters import parse_args
args = parse_args()


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert int(args.num_filters * 1.5) == args.num_filters * 1.5
        self.gru = nn.GRU(
            args.num_filters * 3,
            args.num_filters * 3 if args.long_short_term_method == 'ini' else
            int(args.num_filters * 1.5))

    def forward(self, user, clicked_news_length, history_vector):
        """
        Args:
            user:
                ini: batch_size, num_filters * 3
                con: batch_size, num_filters * 1.5
            clicked_news_length: batch_size,
            history_vector: batch_size, num_history, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """
        clicked_news_length[clicked_news_length == 0] = 1
        # 1, batch_size, num_filters * 3
        if args.long_short_term_method == 'ini':
            packed_history_vector = pack_padded_sequence(history_vector,
                                                         clicked_news_length,
                                                         batch_first=True,
                                                         enforce_sorted=False)
            _, last_hidden = self.gru(packed_history_vector,
                                      user.unsqueeze(dim=0))
            return last_hidden.squeeze(dim=0)
        else:
            packed_history_vector = pack_padded_sequence(history_vector,
                                                         clicked_news_length,
                                                         batch_first=True,
                                                         enforce_sorted=False)
            _, last_hidden = self.gru(packed_history_vector)
            return torch.cat((last_hidden.squeeze(dim=0), user), dim=1)
