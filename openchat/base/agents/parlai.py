from abc import abstractmethod
from typing import Union, Dict
from parlai.core.message import Message
from openchat.base import BaseAgent
import torch


class ParlaiAgent(BaseAgent):

    def __init__(
        self,
        name,
        suffix,
        device,
        maxlen,
        model,
    ):
        super(ParlaiAgent, self).__init__(
            name=name,
            suffix=suffix,
            device=device,
            maxlen=maxlen,
            model=model,
            tokenizer=self.tokenizer,
        )

    def tokenizer(self, message: Union[Message, str]):
        if isinstance(message, str):
            message = Message({
                "text": message,
                "full_text": message,
            })

        history = self.model.build_history()
        history.update_history(message)
        return {"input_ids": history.get_history_vec()}


class ParlaiClassificationAgent(ParlaiAgent):

    @abstractmethod
    def labels(self):
        raise NotImplemented

    def predict(self, text: str, **kwargs):
        message = Message({
            "text": text,
            "full_text": text,
        })

        message["text_vec"] = self.tokenizer(message)['input_ids']
        message["full_text_vec"] = message["text_vec"]
        batch = self.model.batchify([message])

        output = self.model.score(batch)[0].tolist()
        argmax = max(range(len(output)), key=lambda i: output[i])
        return {
            "input": text,
            "output": self.labels()[argmax],
        }


class ParlaiGenerationAgent(ParlaiAgent):

    @torch.no_grad()
    def predict(
        self,
        text,
        method="top_k",
        num_beams=5,
        top_k=20,
        top_p=None,
        no_repeat_ngram_size=4,
        length_penalty: int = 0.65,
    ) -> Dict[str, str]:
        assert method in ["greedy", "beam", "top_k", "nucleus"], \
            "param `method` must be one of ['greedy', 'beam'', 'top_k', 'nucleus']"

        self.model.opt["inference"] = method.replace("_", "")
        self.model.opt["beam_size"] = num_beams
        self.model.opt["topk"] = top_k
        self.model.opt["topp"] = top_p
        self.model.opt["beam-block.ngram"] = no_repeat_ngram_size
        self.model.opt["beam-context-block-ngram"] = no_repeat_ngram_size
        self.model.opt["beam_length_penalty"] = length_penalty

        message = Message({
            "text": text,
            "full_text": text,
        })

        message["text_vec"] = self.tokenizer(message)["input_ids"]
        message["full_text_vec"] = message["text_vec"]
        batch = self.model.batchify([message])

        tokens = self.model._generate(
            batch=batch,
            beam_size=num_beams,
            max_ts=self.maxlen,
        )[0][0][0]

        return {
            "input": text,
            "output": self.model._v2t(tokens.tolist()),
        }
