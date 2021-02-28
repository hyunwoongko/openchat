#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM


class DialoGPT(object):

    def __init__(self, size, device="cuda"):
        """
        Modeling class for Dialo GPT

        Args:
            size (str): model size. must be in ['small', 'medium', 'large']
            device (str): model device, default is 'cuda'

        Notes:
            format of histories:
                self.histories = {
                    0 : {'user': [] , 'bot': []},
                    1 : {'user': [] , 'bot': []},
                    ...more...
                }

            paper (arXiv):
                https://arxiv.org/abs/1911.00536

        Examples:
            >>> # chatting with DialoGPT on terminal mode.
            >>> # The model size must be one of the [small, medium, large].
            >>> # type '/exit' if you want to exit dialogue.
            >>> gpt = DialoGPT(size="large", device="cuda")
            >>> gpt.run()
            user : Hello.
            bot : How are you?
            user : I'm good. how about you?
            bot : Good, you?
            user : Me too.
            bot : That's good.
            user : /exit
            bot : bye.

            >>> # chatting with DialoGPT by user id. (single-turn)
            >>> gpt = DialoGPT(size="large", device="cuda")
            >>> gpt.predict(user_id="USER_ID", text="Hello.")

            >>> # chatting with DialoGPT by user id. (multi-turn)
            >>> while True:
            ...    _in = input('user : ')
            ...    _out = gpt.predict(user_id="USER_ID", text=_in)
            ...    print(f"bot : {_out}")

            >>> # you can check dialogue histories
            >>> gpt.histories
            {
                user_1 : {'user': [] , 'bot': []},
                user_2 : {'user': [] , 'bot': []},
                ...more...
                user_n : {'user': [] , 'bot': []},
            }

        """

        assert size in ['small', 'medium', 'large'], \
            "model size must be in ['small', 'medium', 'large]"

        self.model_name = f"microsoft/DialoGPT-{size}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model = self.model.eval().to(device)

        self.max_context_length = 48
        self.histories = {}
        self.device = device
        self.eos = "<|endoftext|>"

    @torch.no_grad()
    def predict(
            self,
            user_id: Any,
            text: str,
            num_beams: int = 10,  # paper's setting
            top_k: int = 10,  # paper's setting
            top_p: float = None,  # do not use top-p sampling
    ) -> str:
        """
        dialogue with Dialo GPT

        Args:
            user_id (Any): user id
            text (str): user's input text
            num_beams (int): size of beam width
            top_k (int): K for top-K sampling
            top_p (float): P for top-P sampling

        Returns:
            (str): model's next utterance

        """

        torch.cuda.empty_cache()
        input_ids_list: list = []
        num_of_stacked_tokens: int = 0

        if user_id not in self.histories.keys():
            self.clear(user_id)

        user_histories = reversed(self.histories[user_id]['user'])
        bot_histories = reversed(self.histories[user_id]['bot'])

        for user, bot in zip(user_histories, bot_histories):
            user_tokens = self.tokenizer.encode(user, return_tensors='pt')
            bot_tokens = self.tokenizer.encode(bot, return_tensors='pt')
            num_of_stacked_tokens += user_tokens.shape[-1] + bot_tokens.shape[-1]

            if num_of_stacked_tokens <= self.max_context_length:
                input_ids_list.append(bot_tokens)
                input_ids_list.append(user_tokens)

            else:
                break

        input_ids_list = list(reversed(input_ids_list))
        new_input = text + self.eos
        input_tokens = self.tokenizer.encode(new_input, return_tensors='pt')
        input_ids_list.append(input_tokens)

        input_tokens = torch.cat(input_ids_list, dim=-1)
        input_tokens = input_tokens.to(self.device)

        output_ids = self.model.generate(
            input_tokens,
            max_length=1024,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=4,
        )

        next_utterance = self.tokenizer.decode(
            output_ids[:, input_tokens.shape[-1]:][0],
            skip_special_tokens=True,
        )

        self.histories[user_id]['user'].append(text + self.eos)
        self.histories[user_id]['bot'].append(next_utterance + self.eos)

        return next_utterance

    def clear(self, user_id):
        self.histories[user_id] = {'user': [], 'bot': []}

    @staticmethod
    def run():
        while True:
            _in = input("user : ")

            if _in == "/exit":
                print(f"bot : bye.")
                break

            else:
                _out = gpt.predict(user_id="user1", text=_in)
                print(f"bot : {_out}")


if __name__ == '__main__':
    # gpt = DialoGPT("small")
    # gpt = DialoGPT("medium")
    gpt = DialoGPT("large", device="cpu")
    gpt.run()
