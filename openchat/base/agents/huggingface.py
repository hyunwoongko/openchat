import torch

from typing import Dict
from openchat.base import BaseAgent, DecoderLM


class HuggingfaceAgent(BaseAgent):

    @torch.no_grad()
    def predict(
        self,
        text: str,
        method: str = "top_k",
        num_beams: int = 5,
        top_k: int = 20,
        top_p: float = None,
        no_repeat_ngram_size: int = 4,
        length_penalty: int = 0.65,

    ) -> Dict[str, str]:
        """
        Generate utterance.

        Args:
            text (str): input sentence
            num_beams (int): size of beam search
            top_k (int): k value for top-k sampling
            top_p (float): probability for nuclear sampling
            no_repeat_ngram_size (int): no repeat n-gram size

        Returns:
            (Dict[str, str]): user input and generated utterance
        """

        method = method.lower()
        assert method in ["greedy", "beam", "top_k", "nucleus"], \
            "param `method` must be one of ['greedy', 'beam', 'top_k', 'nucleus']"

        if method == "greedy":
            num_beams = 1

        input_ids = self.tokenizer(
            text=text,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        output_ids = self.model.predict(
            input_ids=input_ids,
            num_beams=num_beams,
            top_k=top_k if method == "top_k" else None,
            top_p=top_p if method == "nucleus" else None,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.maxlen * 2,
            length_penalty=length_penalty,
            repetition_penalty=2.0,
            use_cache=True,
        )

        if isinstance(self, DecoderLM):
            # decoder only model
            output_string = self.tokenizer.decode(
                output_ids[:, input_ids.shape[-1]:][0],
                skip_special_tokens=True,
            )
        else:
            # sequence to sequence model
            output_string = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            )

        return {"input": text, "output": output_string}
