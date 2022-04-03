from news_recommendation.model.general.trainer.federated import FederatedModel
from news_recommendation.model.NAML import _NAML


class FedNAML(_NAML, FederatedModel):
    pass
