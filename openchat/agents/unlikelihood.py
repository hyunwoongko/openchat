import importlib
from parlai.core.agents import add_datapath_and_model_args
from parlai.core.build_data import modelzoo_path

from openchat.utils import (
    inherit,
    create_agent_from_opt_file_and_model_class,
)

from openchat.base import (
    ParlaiGenerationAgent,
    Seq2SeqLM,
    ConvAI2Agent,
    WizardOfWikipediaAgent,
)


class UnlikelihoodAgent(ParlaiGenerationAgent, Seq2SeqLM):

    def __init__(self, model, device, maxlen=-1):
        self.check_agent(model)
        maxlen = maxlen if maxlen > 0 else self.default_maxlen()

        if "wizard_of_wikipedia.context_and_label" in model:
            name = "rep_wiki_ctxt_and_label"
        elif "wizard_of_wikipedia.context" in model:
            name = "rep_wiki_ctxt"
        elif "wizard_of_wikipedia.label" in model:
            name = "rep_label_ctxt"
        elif "convai2.context_and_label" in model:
            name = "rep_convai2_ctxt_and_label"
        elif "convai2.context" in model:
            name = "rep_convai2_ctxt"
        elif "convai2.label" in model:
            name = "rep_convai2_label"
        elif "convai2.vocab.alpha.1e-0" in model:
            name = "vocab_alpha1e0"
        elif "convai2.vocab.alpha.1e-1" in model:
            name = "vocab_alpha1e1"
        elif "convai2.vocab.alpha.1e-2" in model:
            name = "vocab_alpha1e2"
        elif "convai2.vocab.alpha.1e-3" in model:
            name = "vocab_alpha1e3"
        elif "eli5.context_and_label" in model:
            name = "rep_eli5_ctxt_and_label"
        elif "eli5.context" in model:
            name = "rep_eli5_ctxt"
        elif "eli5.label" in model:
            name = "rep_eli5_label"
        else:
            raise Exception(f"wrong model: {model}")

        option, model_class = self.set_options(
            name=f"zoo:dialogue_unlikelihood/{name}/model",
            path="projects.dialogue_unlikelihood.agents",
            class_name="RepetitionUnlikelihoodAgent",
        )

        super().__init__(
            device=device,
            name=model,
            maxlen=maxlen,
            suffix="\n",
            model=create_agent_from_opt_file_and_model_class(
                opt=option,
                model_class=model_class,
            ),
        )

        if "wizard_of_wikipedia" in model:
            inherit(self, (WizardOfWikipediaAgent, Seq2SeqLM))
            self.build_wizard_of_wikipedia()

        elif "convai2" in model:
            inherit(self, (ConvAI2Agent, Seq2SeqLM))

    @staticmethod
    def available_models():
        return [
            "unlikelihood.wizard_of_wikipedia.context_and_label",
            "unlikelihood.wizard_of_wikipedia.context",
            "unlikelihood.wizard_of_wikipedia.label",
            "unlikelihood.convai2.context_and_label",
            "unlikelihood.convai2.context",
            "unlikelihood.convai2.label",
            "unlikelihood.convai2.vocab.alpha.1e-0",
            "unlikelihood.convai2.vocab.alpha.1e-1",
            "unlikelihood.convai2.vocab.alpha.1e-2",
            "unlikelihood.convai2.vocab.alpha.1e-3",
            "unlikelihood.eli5.context_and_label",
            "unlikelihood.eli5.context",
            "unlikelihood.eli5.label",
        ]

    def set_options(self, name, path, class_name, device):
        option = {
            "n_image_tokens": 1,
            "n_image_channels": 1,
            "image_fusion_type": "late",
        }
        add_datapath_and_model_args(option)
        datapath = option.get('datapath')
        option['model_file'] = modelzoo_path(datapath, name)
        option["override"] = {
            "no_cuda": False if "cuda" in device else True,
        }
        my_module = importlib.import_module(path)
        model_class = getattr(my_module, class_name)
        return option, model_class

    @staticmethod
    def default_maxlen():
        return 128
