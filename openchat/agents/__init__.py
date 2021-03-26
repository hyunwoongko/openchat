from openchat.agents.base import BaseAgent, EncoderLM, DecoderLM, Seq2SeqLM
from openchat.agents.huggingface import HuggingfaceAgent
from openchat.agents.parlai import ParlaiAgent, ParlaiGenerationAgent, ParlaiClassificationAgent
from openchat.agents.convai2 import ConvAI2Agent
from openchat.agents.wow import WizardOfWikipediaAgent

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
]
