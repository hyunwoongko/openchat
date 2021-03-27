import json
import os

from random import choice
from typing import Dict
from parlai.core.agents import create_agent, add_datapath_and_model_args
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.tasks.wizard_of_wikipedia.build import build
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import KnowledgeRetrieverAgent
from openchat.base import ParlaiGenerationAgent


def load_topics():
    opt = {}
    add_datapath_and_model_args(opt)
    build(opt)

    topics_path = os.path.join(
        opt['datapath'],
        'wizard_of_wikipedia',
        'topic_splits.json',
    )

    return sorted(json.load(open(topics_path, 'rb'))["train"])


def create_retriever():
    parser = ParlaiParser(False, False)
    KnowledgeRetrieverAgent.add_cmdline_args(parser)
    parser.set_params(
        model='projects:wizard_of_wikipedia:knowledge_retriever',
        add_token_knowledge=False,
    )
    knowledge_opt = parser.parse_args([])
    return create_agent(knowledge_opt)


class WizardOfWikipediaAgent(ParlaiGenerationAgent):

    def __init__(
        self,
        name,
        suffix,
        device,
        maxlen,
        model,
    ):
        super().__init__(
            name=name,
            suffix=suffix,
            device=device,
            maxlen=maxlen,
            model=model,
        )

        self.build_wizard_of_wikipedia()

    def build_wizard_of_wikipedia(self):
        self.TOKEN_KNOWLEDGE = '__knowledge__'
        self.TOKEN_END_KNOWLEDGE = '__endknowledge__'
        self.topic_list = load_topics()
        self.knowledge_retriever = create_retriever()
        self.chosen_topic = None

    def available_topics(self):
        return self.topic_list

    def set_topic(self, topic):
        if topic.lower() == "random":
            self.chosen_topic = choice(self.topic_list)

        else:
            assert topic in self.topic_list, \
                f"Wrong topic: {topic}, You can check available topic using `available_topics()`"
            self.chosen_topic = topic

    def clear_topic(self):
        self.chosen_topic = None

    def retrieve_knowledge(self, text):
        message = Message({
            "id": "local_human",
            "text": self.chosen_topic + self.suffix + text,
            "chosen_topic": self.chosen_topic,
            "episode_done": False,
            "label_candidates": None,
        })

        self.knowledge_retriever.observe(message)
        knowledge = self.knowledge_retriever.act()["checked_sentence"]
        knowledge = self.TOKEN_KNOWLEDGE + knowledge + self.TOKEN_END_KNOWLEDGE
        return knowledge + self.suffix + text

    def predict(
        self,
        text,
        method="beam",
        num_beams=5,
        top_k=None,
        top_p=None,
        no_repeat_ngram_size=4,
        length_penalty=0.65,
    ) -> Dict[str, str]:
        if not self.chosen_topic:
            raise Exception(
                "topic isn't selected. "
                "please call `set_topic(topic: str)` to select topic for wizard of wikipedia task"
            )

        return super().predict(
            text=text,
            method=method,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
        )
