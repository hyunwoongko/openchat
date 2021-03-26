import json
import os

from random import choice
from typing import Dict
from parlai.core.agents import create_agent, add_datapath_and_model_args
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.tasks.wizard_of_wikipedia.build import build
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import KnowledgeRetrieverAgent
from openchat.agents import ParlaiGenerationAgent


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


class WizardOfWikipediaAgent(ParlaiGenerationAgent):

    TOKEN_KNOWLEDGE = '__knowledge__'
    TOKEN_END_KNOWLEDGE = '__endknowledge__'
    topic_list = load_topics()
    chosen_topic = None
    knowledge_retriever = None

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

    def create_retriever(self):
        parser = ParlaiParser(False, False)
        KnowledgeRetrieverAgent.add_cmdline_args(parser)
        parser.set_params(
            model='projects:wizard_of_wikipedia:knowledge_retriever',
            add_token_knowledge=False,
        )
        knowledge_opt = parser.parse_args([])
        return create_agent(knowledge_opt)

    def retrieve_knowledge(self, text):
        if not self.knowledge_retriever:
            self.knowledge_retriever = self.create_retriever()

        message = Message({
            "id": "local_human",
            "text": self.chosen_topic + self.suffix + text,
            "chosen_topic": self.chosen_topic,
            "episode_done": False,
            "label_candidates": None,
        })

        self.knowledge_retriever.observe(message)
        knowledge = self.knowledge_retriever.act()["checked_sentence"]
        knowledge = self.TOKEN_KNOWLEDGE + knowledge + self.TOKEN_END_KNOWLEDGE + self.suffix

        if self.suffix not in text:
            knowledge += self.chosen_topic + self.suffix

        return knowledge + text

    def predict(
        self,
        text,
        method="beam",
        num_beams=5,
        top_k=None,
        top_p=None,
        no_repeat_ngram_size=4,
    ) -> Dict[str, str]:

        return super().predict(
            text=self.retrieve_knowledge(text),
            method=method,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
