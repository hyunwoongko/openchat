from openchat.agents import WizardOfWikipediaAgent, Seq2SeqLM


class WizardOfWikipedia(WizardOfWikipediaAgent, Seq2SeqLM):

    def __init__(self, model, device, maxlne=256):
        model = self.check_model(model)

    def available_models(self):
        return [

        ]

