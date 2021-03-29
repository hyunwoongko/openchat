from parlai.core.agents import add_datapath_and_model_args, create_agent_from_opt_file
from parlai.core.build_data import modelzoo_path

from openchat.base import ParlaiGenerationAgent, Seq2SeqLM


class RedditAgent(ParlaiGenerationAgent, Seq2SeqLM):

    def __init__(self, model: str, device: str, maxlen) -> None:
        self.check_agent(model)
        maxlen = maxlen if maxlen > 0 else self.default_maxlen()

        if "xlarge" in model:
            size = "3B"
        elif "xxlarge" in model:
            size = "9B"
        else:
            raise Exception("wrong model")

        option = self.set_options(
            name=f"zoo:blender/reddit_{size}/model",
            device=device,
        )

        super().__init__(
            name=model,
            suffix="\n",
            device=device,
            maxlen=maxlen,
            model=create_agent_from_opt_file(option),
        )

    @staticmethod
    def available_models():
        return [
            "reddit.xlarge",
            "reddit.xxlarge",
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
