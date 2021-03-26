from parlai.core.agents import create_agent_from_model_file, add_datapath_and_model_args
from parlai.core.build_data import modelzoo_path
from projects.dialogue_unlikelihood.agents import RepetitionUnlikelihoodAgent

from openchat.utils import inherit
from openchat.agents import (
    ParlaiGenerationAgent,
    Seq2SeqLM,
    ConvAI2Agent,
    WizardOfWikipediaAgent,
)


class Unlikelihood(ParlaiGenerationAgent, Seq2SeqLM):

    def __init__(self, model, device, maxlen=256):
        self.check_model(model)

        if "wizard_of_wikipedia.context_and_label" in model:
            name = "zoo:dialogue_unlikelihood/rep_wiki_ctxt_and_label/model"
        elif "wizard_of_wikipedia.context" in model:
            name = "zoo:dialogue_unlikelihood/rep_wiki_ctxt"
        elif "wizard_of_wikipedia.label" in model:
            name = "zoo:dialogue_unlikelihood/rep_label_ctxt"
        elif "convai2.context_and_label" in model:
            name = "zoo:dialogue_unlikelihood/rep_convai2_ctxt_and_label/model"
        elif "convai2.context" in model:
            name = "zoo:dialogue_unlikelihood/rep_convai2_ctxt/model"
        elif "convai2.label" in model:
            name = "zoo:dialogue_unlikelihood/rep_convai2_label/model"
        elif "convai2.vocab.alpha.1e-0" in model:
            name = "zoo:dialogue_unlikelihood/vocab_alpha1e0/model"
        elif "convai2.vocab.alpha.1e-1" in model:
            name = "zoo:dialogue_unlikelihood/vocab_alpha1e1/model"
        elif "convai2.vocab.alpha.1e-2" in model:
            name = "zoo:dialogue_unlikelihood/vocab_alpha1e2/model"
        elif "convai2.vocab.alpha.1e-3" in model:
            name = "zoo:dialogue_unlikelihood/vocab_alpha1e3/model"
        elif "eli5.context_and_label" in model:
            name = "zoo:dialogue_unlikelihood/rep_eli5_ctxt_and_label/model"
        elif "eli5.context" in model:
            name = "zoo:dialogue_unlikelihood/rep_eli5_ctxt/model"
        elif "eli5.label" in model:
            name = "zoo:dialogue_unlikelihood/rep_eli5_label/model"
        else:
            raise Exception(f"wrong model: {model}")

        opt = {}
        add_datapath_and_model_args(opt)
        opt['model_file'] = modelzoo_path(opt.get('datapath'), name)
        opt['no_cuda'] = True if device == "cpu" else False
        opt["history_size"] = -1
        opt["truncate"] = -1
        opt["rank_candidates"] = False

        super(Unlikelihood, self).__init__(
            device=device,
            name=model,
            maxlen=maxlen,
            prefix="",
            suffix="\n",
            model=RepetitionUnlikelihoodAgent(opt),
        )

        if "wizard_of_wikipedia" in name:
            inherit(self, (WizardOfWikipediaAgent, Seq2SeqLM))

        elif "convai2" in name:
            inherit(self, (ConvAI2Agent, Seq2SeqLM))

    def available_models(self):
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


if __name__ == '__main__':
    unl = Unlikelihood("unlikelihood.convai2.context_and_label", "cpu")
    output = unl.predict("hello.")
    print(output)
