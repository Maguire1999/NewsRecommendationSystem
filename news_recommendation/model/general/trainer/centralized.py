import torch
import torch.nn as nn
from news_recommendation.parameters import parse_args
args = parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CentralizedModelTrainer():
    def init(self, model):
        if args.loss == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(),
                                              lr=args.learning_rate)
        elif args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(model.parameters(),
                                             lr=args.learning_rate)
        else:
            raise NotImplementedError

    def backward(self, y_pred):
        if args.loss == 'BCE':
            y = torch.cat(
                (torch.ones(y_pred.size(0), 1),
                 torch.zeros(y_pred.size(0), args.negative_sampling_ratio)),
                dim=1).to(device)
        else:
            y = torch.zeros(y_pred.size(0)).long().to(device)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
