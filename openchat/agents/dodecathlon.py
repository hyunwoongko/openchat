from parlai.core.build_data import modelzoo_path
from parlai.core.agents import (
    add_datapath_and_model_args,
    create_agent_from_opt_file,
)
from openchat.utils import inherit
from openchat.base import (
    ParlaiGenerationAgent,
    ConvAI2Agent,
    WizardOfWikipediaAgent,
    Seq2SeqLM,
)


class DodecathlonAgent(ParlaiGenerationAgent, Seq2SeqLM):

    def __init__(self, model, device, maxlen):
        model = self.check_agent(model)
        maxlen = maxlen if maxlen > 0 else self.default_maxlen()

        model = model + "_ft" if model != "all_tasks_mt" else model
        name = f"zoo:dodecadialogue/{model.split('.')[-1]}/model"
        option = self.set_options(name, device)

        super().__init__(
            name=model,
            suffix="\n",
            device=device,
            maxlen=maxlen,
            model=create_agent_from_opt_file(option),
        )

        if "wizard_of_wikipedia" in name:
            inherit(self, (WizardOfWikipediaAgent, Seq2SeqLM))
            self.build_wizard_of_wikipedia()

        elif "convai2" in name:
            inherit(self, (ConvAI2Agent, Seq2SeqLM))

    @staticmethod
    def available_models():
        return [
            "dodecathlon.all_tasks_mt",
            "dodecathlon.convai2",
            "dodecathlon.wizard_of_wikipedia",
            "dodecathlon.empathetic_dialogues"
            "dodecathlon.eli5",
            "dodecathlon.reddit",
            "dodecathlon.twitter",
            "dodecathlon.ubuntu",
            "dodecathlon.image_chat",
            "dodecathlon.cornell_movie",
            "dodecathlon.light_dialog",
            "dodecathlon.daily_dialog",
        ]

    def set_options(self, name, device):
        option = {
            "n_image_tokens": 1,
            "n_image_channels": 1,
            "image_fusion_type": "late",
        }

        add_datapath_and_model_args(option)
        datapath = option.get("datapath")
        option['model_file'] = modelzoo_path(datapath, name)
        option["override"] = {
            "no_cuda": False if "cuda" in device else True,
        }
        return option

    @staticmethod
    def default_maxlen():
        return 128
