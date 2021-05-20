from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.build_data import modelzoo_path
from parlai.utils.safety import OffensiveStringMatcher
from parlai.core.agents import add_datapath_and_model_args, create_agent_from_opt_file, create_agent
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
        self.agent = self._create_safety_model(
            "zoo:dialogue_safety/single_turn/model",
            device=device,
        )
        self.model = self.agent.model

    def _create_safety_model(self, custom_model_file, device):
        from parlai.core.params import ParlaiParser

        parser = ParlaiParser(False, False)
        TransformerClassifierAgent.add_cmdline_args(parser, partial_opt=None)
        parser.set_params(
            model='transformer/classifier',
            model_file=custom_model_file,
            print_scores=True,
            data_parallel=False,
        )
        safety_opt = parser.parse_args([])
        safety_opt["override"]["no_cuda"] = False if "cuda" in device else True
        return create_agent(safety_opt, requireModelExists=True)

    def contains_offensive_language(self, text):
        """
        Returns the probability that a message is safe according to the classifier.
        """
        act = {'text': text, 'episode_done': True}
        self.agent.observe(act)
        response = self.agent.act()['text']
        pred_class, prob = [x.split(': ')[-1] for x in response.split('\n')]
        pred_not_ok = self.labels()[0 if pred_class == "__ok__" else 1]
        return pred_not_ok

    def labels(self):
        return ["non-offensive", "offensive"]

    def predict(self, text, method="both", **kwargs):
        assert method in ["both", "string-match", "bert"], \
            "param method must be one of ['both', 'string-match', 'bert']"

        if method == "string-match":
            return {
                "input":
                    text,
                "output":
                    "offensive"
                    if text in self.string_matcher else "non-offensive",
            }

        if method == "bert":
            return {
                "input": text,
                "output": self.contains_offensive_language(text),
            }

        if method == "both":
            if text in self.string_matcher:
                return {
                    "input": text,
                    "output": "offensive",
                }

            return {
                "input": text,
                "output": self.contains_offensive_language(text),
            }

    @staticmethod
    def available_models():
        return ['safety.offensive']

    @staticmethod
    def default_maxlen():
        return 128


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
        return 128

    def set_options(self, name, device):
        option = {}

        add_datapath_and_model_args(option)
        datapath = option.get("datapath")
        option['model_file'] = modelzoo_path(datapath, name)
        option = {
            "no_cuda": False if "cuda" in device else True,
        }
        return option
