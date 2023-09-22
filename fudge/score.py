from abc import ABC,abstractmethod

class ScoreModule(ABC):
    def __init__(self, model, device, condition_lambda) -> None:
        self.model = model
        self.device = device
        self.condition_lambda = condition_lambda
    
    @abstractmethod
    def score(self):
        pass