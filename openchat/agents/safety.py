from parlai.core.agents import create_agent_from_model_file
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier
from openchat.base import EncoderLM, ParlaiAgent, ParlaiClassificationAgent


class SafetyAgent(ParlaiAgent, EncoderLM):

    def __init__(self, model, device, maxlen):
        model = self.check_agent(model)
        maxlen = maxlen if maxlen > 0 else self.default_maxlen()

        if "offensive" in model:
            self.model = OffensiveClassifier(
                model=model,
                device=device,
                maxlen=maxlen,
            )

        elif "sensitive" in model:
            self.model = SensitiveClassifier(
                model=model,
                device=device,
                maxlen=maxlen,
            )
        else:
            raise Exception("wrong model")

        super(SafetyAgent, self).__init__(
            device=device,
            maxlen=maxlen,
            model=self.model,
            suffix="",
            name=model,
        )

    def predict(self, text, **kwargs):
        return self.model.predict(text)

    @staticmethod
    def available_models():
        return [
            "safety.offensive",
            "safety.sensitive",
        ]

    @staticmethod
    def default_maxlen():
        return 512


class OffensiveClassifier(ParlaiClassificationAgent, EncoderLM):

    def __init__(self, model, device, maxlen):
        self.string_matcher = OffensiveStringMatcher()
        self.model = OffensiveLanguageClassifier()
        super(OffensiveClassifier, self).__init__(
            device=device,
            maxlen=maxlen,
            model=self.model,
            prefix="",
            suffix="",
            name=model,
        )

    def labels(self):
        return ["offensive", "non-offensive"]

    def predict(self, text, **kwargs):
        if text in self.string_matcher:
            return {
                "input": text,
                "output": "offensive",
            }

        return {
            "input": text,
            "output": self.labels()[text in self.model],
        }

    @staticmethod
    def available_models():
        return []

    @staticmethod
    def default_maxlen():
        return 512


class SensitiveClassifier(ParlaiClassificationAgent, EncoderLM):

    def __init__(self, model, device, maxlen):
        self.model = create_agent_from_model_file(
            "zoo:sensitive_topics_classifier/model")
        super(SensitiveClassifier, self).__init__(
            device=device,
            maxlen=maxlen,
            model=self.model,
            prefix="",
            suffix="",
            name=model,
        )

    def labels(self):
        return [
            "drugs",
            "politics",
            "religion",
            "medical_advice",
            "dating",
            "safe",
        ]

    @staticmethod
    def available_models():
        return []

    @staticmethod
    def default_maxlen():
        return 512
