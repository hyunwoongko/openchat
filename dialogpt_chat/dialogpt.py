#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class DialoGPT(object):

    def __init__(self, size, device, max_context_length=48):
        """
        Modeling class for Dialo GPT

        Args:
            size (str): model size. must be one of ['small', 'medium', 'large']
            device (str): model device. should be one of ['cpu', 'cuda', 'cuda:n']
            max_context_length (int): max context laength (number of input history tokens)

        Notes:
            format of histories:
                self.histories = {
                    user_1 : {'user': [] , 'bot': []},
                    user_2 : {'user': [] , 'bot': []},
                    ...more...
                    user_n : {'user': [] , 'bot': []},
                }

            paper (arXiv):
                https://arxiv.org/abs/1911.00536

        Examples:
            >>> # chatting with DialoGPT on terminal mode.
            >>> # The model size must be one of the [small, medium, large].
            >>> # type '/exit' if you want to exit dialogue.
            >>> # type '/clear' if you want to clear all histories
            >>> gpt = DialoGPT(size="large", device="cuda")
            >>> gpt.run()
            user : Hello.
            bot : How are you?
            user : I'm great. it is a nice day.
            bot : That's good.
            user : Who is CEO of Apple?
            bot : Steve Jobs.
            user : /clear
            bot : history cleared.
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

            >>> # you can clear all dialogue histories
            >>> gpt.clear(user_id="USER_ID")

        """

        assert size in ['small', 'medium', 'large'], \
            "model size must be one of ['small', 'medium', 'large]"

        self.model_name = f"microsoft/DialoGPT-{size}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model = self.model.eval().to(device)

        self.max_context_length = max_context_length
        self.histories = {}
        self.device = device
        self.eos = "<|endoftext|>"

    @torch.no_grad()
    def predict(
            self,
            user_id: str,
            text: str,
            num_beams: int = 10,  # paper's setting
            top_k: int = 10,  # paper's setting
            top_p: float = None,  # do not use top-p sampling
    ) -> str:
        """
        dialogue with Dialo GPT

        Args:
            user_id (str): user id
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

    def run(self):
        while True:
            _in = input("user : ")

            if _in == "/exit":
                print(f"bot : bye.")
                break

            elif _in == "/clear":
                print(f"bot : history cleared.")
                self.clear("user_id")

            else:
                _out = self.predict(user_id="user_id", text=_in)
                print(f"bot : {_out}")
