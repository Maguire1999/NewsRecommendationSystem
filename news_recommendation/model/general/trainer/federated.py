import torch
import torch.nn as nn

from news_recommendation.shared import args, device


class FederatedModelTrainer():
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
        elif args.loss == 'CE':
            y = torch.zeros(y_pred.size(0)).long().to(device)
        else:
            raise NotImplementedError

        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
