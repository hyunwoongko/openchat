import torch
from openchat.models.base_model import BaseModel
from transformers import (
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)


class BlenderBot(BaseModel):

    def __init__(self, size, env, device, max_context_length):
        """
        The Blender chatbot model was proposed in Recipes for building an open-domain chatbot Stephen Roller,
        Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith,
        Y-Lan Boureau, Jason Weston on 30 Apr 2020.

        Args:
            size (str): model size
            env (BaseEnv): dialogue environment
            device (str): device (one of ['CPU', 'CUDA', 'CUDA:N']
            max_context_length (int): max history context length
                (it means that length of input context tokens)
        """

        assert size in ['small', 'medium', 'large', 'xlarge'], \
            "model size must be one of ['small', 'medium', 'large', 'xlarge']"

        if size == "small":
            super().__init__("facebook/blenderbot_small-90M", env)
            self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(
                self.name).to(device)
            self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(self.name)
        else:
            if size == "medium":
                super().__init__("facebook/blenderbot-400M-distill", env)
            elif size == "large":
                super().__init__("facebook/blenderbot-1B-distill", env)
            elif size == "xlarge":
                super().__init__("facebook/blenderbot-3B", env)

            self.model = BlenderbotForConditionalGeneration.from_pretrained(
                self.name).to(device)
            self.tokenizer = BlenderbotTokenizer.from_pretrained(self.name)

        self.device = device.lower()
        self.max_context_length = max_context_length
        self.eos = "</s> <s>"

    @torch.no_grad()
    def predict(
        self,
        user_id: str,
        text: str,
        num_beams: int = 5,
        top_k: int = 1,
        top_p: float = None,
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

        if user_id not in self.env.histories.keys():
            self.env.clear(user_id, text)

        user_histories = reversed(self.env.histories[user_id]['user'])
        bot_histories = reversed(self.env.histories[user_id]['bot'])

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
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=4,
        )[0]

        next_utterance = self.tokenizer.decode(
            output_ids.tolist(),
            skip_special_tokens=True,
        ).replace("Ġ", "").replace("  ", "")

        self.env.histories[user_id]['user'].append(text + self.eos)
        self.env.histories[user_id]['bot'].append(next_utterance + self.eos)

        return next_utterance
