import torch

from openchat.base import HuggingfaceAgent


class PromptAgent(HuggingfaceAgent):

    def add_prompt(self, histories, user_id, text):
        histories[user_id]["prefix"].append(text)

    def clear_prompt(self, histories, user_id):
        histories[user_id]["prefix"] = [
            pf for pf in histories[user_id]["prefix"]
        ]

    @torch.no_grad()
    def predict(
        self,
        text,
        person_1: str,
        person_2: str,
        method: str = "top_k",
        top_k: int = 20,
        top_p: float = None,
        num_beams: int = 6,
        num_beam_groups=2,
        length_penalty=0.7,
        diverse_penalty=1.5,
        no_repeat_ngram_size=4,
        **kwargs,
    ):

        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
        )["input_ids"]

        output_ids = self.model.generate(
            input_ids=input_ids.to(self.device),
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            length_penalty=length_penalty,
            repetition_penalty=2.0,
            top_k=top_k if method == "top_k" else None,
            top_p=top_p if method == "nucleus" else None,
            no_repeat_ngram_size=no_repeat_ngram_size,
            diverse_penalty=diverse_penalty,
            use_cache=True,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=input_ids.size()[-1] + self.maxlen // 2,
            **kwargs,
        )

        turn_escapes = [
            f"{person_1}:",
            f"{person_2}:",
            f"{person_1.lower()}:",
            f"{person_2.lower()}:",
            f"{person_1.upper()}:",
            f"{person_2.upper()}:",
        ]

        generated_text = self.tokenizer.decode(
            output_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True,
        )

        for escape in turn_escapes:
            generated_text = generated_text.replace(escape, "\n")

        return {
            "input": text,
            "output": generated_text.split("\n")[0],
        }
