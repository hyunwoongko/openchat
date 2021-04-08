from abc import abstractmethod
from typing import Dict, List, Union
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

    def tokenizer(self, message: Union[str, List[str]], padding=False):
        if isinstance(message, str):
            return {"input_ids": self.model.dict.txt2vec(message)}
        elif isinstance(message, list):
            if all(isinstance(s, str) for s in message):
                tokens = [self.model.dict.txt2vec(s) for s in message]

                if padding:
                    tokens = self.model._pad_tensor(tokens)[0]

                return {"input_ids": tokens}
            else:
                raise TypeError(
                    f"type error: {type(message)}, input type must be one of [str, List[str]]"
                )
        else:
            raise TypeError(
                f"type error: {type(message)}, input type must be one of [str, List[str]]"
            )


class ParlaiClassificationAgent(ParlaiAgent):

    @abstractmethod
    def labels(self):
        raise NotImplemented

    def predict(self, text: str, **kwargs):
        message = self.tokenizer(text)
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

        vector = self.tokenizer(text)["input_ids"]
        message["text_vec"] = vector
        message["full_text_vec"] = vector

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
