from openchat.base.agents.base import BaseAgent, EncoderLM, DecoderLM, Seq2SeqLM, SingleTurn
from openchat.base.agents.huggingface import HuggingfaceAgent
from openchat.base.agents.parlai import ParlaiAgent, ParlaiGenerationAgent, ParlaiClassificationAgent
from openchat.base.agents.convai2 import ConvAI2Agent
from openchat.base.agents.wow import WizardOfWikipediaAgent

__all__ = [
    BaseAgent,
    HuggingfaceAgent,
    ParlaiAgent,
    ParlaiGenerationAgent,
    ParlaiClassificationAgent,
    ConvAI2Agent,
    WizardOfWikipediaAgent,
    EncoderLM,
    DecoderLM,
    Seq2SeqLM,
    SingleTurn,

]
