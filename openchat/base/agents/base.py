import parlai.utils.logging as logging

from abc import ABC, abstractmethod
from typing import Dict

logging.disable()


# marker interface
class EncoderLM:
    pass


# marker interface
class DecoderLM:
    pass


# marker interface
class Seq2SeqLM:
    pass


# marker interface
class SingleTurn:
    pass


class BaseAgent(ABC):

    def __init__(self, name, suffix, device, maxlen, model, tokenizer):
        self.name = name
        self.suffix = suffix
        self.device = device
        self.maxlen = maxlen
        self.model = model
        self.tokenizer = tokenizer

    def check_agent(self, model) -> str:
        model = model.lower()
        available_models = self.available_models()

        assert model in available_models, \
            f"param `model` must be one of {available_models}"

        return model

    @abstractmethod
    def predict(self, text: str, **kwargs) -> Dict[str, str]:
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def default_maxlen():
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def available_models():
        raise NotImplemented
