from parlai.core.build_data import modelzoo_path
from parlai.core.agents import (
    add_datapath_and_model_args,
    create_agent_from_opt_file,
)
from openchat.utils import inherit
from openchat.agents import (
    ParlaiGenerationAgent,
    ConvAI2Agent,
    WizardOfWikipediaAgent,
    Seq2SeqLM,
)


class Dodecathlon(ParlaiGenerationAgent, Seq2SeqLM):

    def __init__(self, model, device, maxlen=256):
        model = self.check_model(model)
        model = model + "_ft" if model != "all_task_mt" else model
        name = f"zoo:dodecadialogue/{model.split('.')[-1]}/model"

        opt = {}
        add_datapath_and_model_args(opt)
        opt["n_image_tokens"] = 1
        opt["n_image_channels"] = 1
        opt["image_fusion_type"] = "late"
        opt['model_file'] = modelzoo_path(opt.get('datapath'), name)

        super().__init__(
            name=model,
            prefix="",
            suffix="\n",
            device=device,
            maxlen=maxlen,
            model=create_agent_from_opt_file(opt),
        )

        if "wizard_of_wikipedia" in name:
            inherit(self, (WizardOfWikipediaAgent, Seq2SeqLM))

        elif "convai2" in name:
            inherit(self, (ConvAI2Agent, Seq2SeqLM))

    def available_models(self):
        return [
            "dodecathlon.all_task_mt",
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
