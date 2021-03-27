import unittest
import logging
from openchat.agents.blender import BlenderGenerationAgent
from openchat.agents.dialogpt import DialoGPTAgent
from openchat.agents.dodecathlon import DodecathlonAgent
from openchat.agents.reddit import RedditAgent
from openchat.agents.offensive import SafetyAgent
from openchat.agents.unlikelihood import UnlikelihoodAgent
from openchat.agents.wow import WizardOfWikipediaGenerationAgent
from openchat.base import WizardOfWikipediaAgent


class ModelTester(unittest.TestCase):

    def model_unittest(self, model_name, model_class):
        model = model_class(model=model_name, device="cpu")
        if isinstance(model, WizardOfWikipediaAgent):
            model.set_topic("Guitar")

        output = model.predict("hello.")
        logging.info(f"{model} testing is success: {output}")
        self.assertIsInstance(output, dict)

    def test_blender(self):
        for model in BlenderGenerationAgent.available_models():
            self.model_unittest(model, BlenderGenerationAgent)

    def test_dialogpt(self):
        for model in DialoGPTAgent.available_models():
            self.model_unittest(model, DialoGPTAgent)

    def test_dodecathlon(self):
        for model in DodecathlonAgent.available_models():
            self.model_unittest(model, DodecathlonAgent)

    def test_reddit(self):
        for model in RedditAgent.available_models():
            self.model_unittest(model, RedditAgent)

    def test_safety(self):
        for model in SafetyAgent.available_models():
            self.model_unittest(model, SafetyAgent)

    def test_unlikelihood(self):
        for model in UnlikelihoodAgent.available_models():
            self.model_unittest(model, UnlikelihoodAgent)

    def test_wow(self):
        for model in WizardOfWikipediaGenerationAgent.available_models():
            self.model_unittest(model, WizardOfWikipediaGenerationAgent)
