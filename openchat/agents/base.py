from abc import ABC, abstractmethod
from typing import Dict


class EncoderLM:
    use_prefix = False
    use_suffix = False


class DecoderLM:
    use_prefix = True
    use_suffix = True


class Seq2SeqLM:
    use_prefix = True
    use_suffix = False


class BaseAgent(ABC):

    def __init__(self, name, prefix, suffix, device, maxlen, model, tokenizer):
        self.name = name
        self.prefix = prefix
        self.suffix = suffix
        self.device = device
        self.maxlen = maxlen
        self.model = model
        self.tokenizer = tokenizer

    def add_prefix(self, text):
        return self.prefix + text

    def add_suffix(self, text):
        return text + self.suffix

    @abstractmethod
    def predict(self, text: str, **kwargs) -> Dict[str, str]:
        raise NotImplemented

    @abstractmethod
    def available_models(self):
        raise NotImplemented

    def check_model(self, model) -> str:
        model = model.lower()
        available_models = self.available_models()

        assert model in available_models, \
            f"param `model` must be one of {available_models}"

        return model