from parlai.core.agents import create_agent_from_model_file
from openchat.base import WizardOfWikipediaAgent, Seq2SeqLM


class WizardOfWikipediaGenerationAgent(WizardOfWikipediaAgent, Seq2SeqLM):

    def __init__(self, model, device, maxlen=-1):
        model = self.check_agent(model)
        maxlen = maxlen if maxlen > 0 else self.default_maxlen()

        if "end2end_generator" in model:
            name = "end2end_generator"
        elif "full_dialogue_retrieval_model" in model:
            name = "full_dialogue_retrieval_model"
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
        return 256
