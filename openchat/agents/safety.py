from parlai.core.build_data import modelzoo_path
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier
from parlai.core.agents import add_datapath_and_model_args, create_agent_from_opt_file
from openchat.base import ParlaiClassificationAgent, EncoderLM, SingleTurn


class OffensiveAgent(ParlaiClassificationAgent, EncoderLM, SingleTurn):

    def __init__(self, model, device, maxlen):
        super(OffensiveAgent, self).__init__(
            device=device,
            maxlen=maxlen,
            model=None,
            suffix="",
            name=model,
        )
        self.string_matcher = OffensiveStringMatcher()
        self.agent = OffensiveLanguageClassifier()
        self.agent.model.opt["no_cuda"] = \
            True if "cuda" in device else False
        self.model = self.agent.model

    def labels(self):
        return ["non-offensive", "offensive"]

    def predict(self, text, **kwargs):
        if text in self.string_matcher:
            return {
                "input": text,
                "output": "offensive",
            }

        return {
            "input": text,
            "output": self.labels()[int(text in self.agent)],
        }

    @staticmethod
    def available_models():
        return ['safety.offensive']

    @staticmethod
    def default_maxlen():
        return 512


class SensitiveAgent(ParlaiClassificationAgent, EncoderLM, SingleTurn):

    def __init__(self, model, device, maxlen):
        option = self.set_options(
            name="zoo:sensitive_topics_classifier/model",
            device=device,
        )

        super(SensitiveAgent, self).__init__(
            device=device,
            maxlen=maxlen,
            model=create_agent_from_opt_file(option),
            suffix="",
            name=model,
        )

    def labels(self):
        return [
            "safe",
            "politics",
            "religion",
            "medical_advice",
            "dating",
            "drugs",
        ]

    @staticmethod
    def available_models():
        return ['safety.sensitive']

    @staticmethod
    def default_maxlen():
        return 512

    def set_options(self, name, device):
        option = {
            "no_cuda": True if "cuda" in device else False,
        }

        add_datapath_and_model_args(option)
        datapath = option.get("datapath")
        option['model_file'] = modelzoo_path(datapath, name)
        return option
