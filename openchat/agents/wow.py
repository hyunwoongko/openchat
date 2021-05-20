from parlai.core.agents import create_agent_from_model_file, add_datapath_and_model_args
from parlai.core.build_data import modelzoo_path

from openchat.base import WizardOfWikipediaAgent, Seq2SeqLM


class WizardOfWikipediaGenerationAgent(WizardOfWikipediaAgent, Seq2SeqLM):

    def __init__(self, model, device, maxlen=-1):
        model = self.check_agent(model)
        maxlen = maxlen if maxlen > 0 else self.default_maxlen()

        if "end2end_generator" in model:
            name = "end2end_generator"
        else:
            raise Exception("wrong model")

        super().__init__(
            name=model,
            suffix="\n",
            device=device,
            maxlen=maxlen,
            model=create_agent_from_model_file(
                f"zoo:wizard_of_wikipedia/{name}/model"),
        )

    @staticmethod
    def available_models():
        return [
            "wizard_of_wikipedia.end2end_generator",
        ]

    @staticmethod
    def default_maxlen():
        return 128

    def set_options(self, name, device):
        option = {}

        add_datapath_and_model_args(option)
        datapath = option.get("datapath")
        option['model_file'] = modelzoo_path(datapath, name)
        option["override"] = {
            "no_cuda": False if "cuda" in device else True,
        }
        return option
