from parlai.core.agents import create_agent_from_model_file
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier
from openchat.agents import EncoderLM, ParlaiAgent, ParlaiClassificationAgent


class Safety(ParlaiAgent, EncoderLM):

    def __init__(self, model, device, maxlen=-1):
        model = self.check_model(model)

        if model == "offensive":
            self.model = OffensiveClassifier(device=device)
        elif model == "sensitive":
            self.model = SensitiveClassifier(device=device)

        super(Safety, self).__init__(
            device=device,
            maxlen=maxlen,
            model=self.model,
            prefix="",
            suffix="",
            name=model,
        )

    def predict(self, text, **kwargs):
        return self.model.predict(text)

    def available_models(self):
        return [
            "safety.offensive",
            "safety.sensitive",
        ]


class OffensiveClassifier(ParlaiClassificationAgent, EncoderLM):

    def __init__(self, model, device, maxlen=-1):
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

    def available_models(self):
        return []


class SensitiveClassifier(ParlaiClassificationAgent, EncoderLM):

    def __init__(self, model, device, maxlen=-1):
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

    def available_models(self):
        return []
