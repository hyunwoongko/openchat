import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openchat.base import DecoderLM
from openchat.base.agents.prompt import PromptAgent


class GPTNeoAgent(PromptAgent, DecoderLM):

    def __init__(self, model, device, maxlen):
        model = self.check_agent(model)
        maxlen = maxlen if maxlen > 0 else self.default_maxlen()

        if "small" in model:
            size = "125M"
        elif "medium" in model:
            size = "350M"
        elif "large" in model:
            size = "1.3B"
        elif "xlarge" in model:
            size = "2.7B"
        else:
            raise Exception("wrong model")

        name = f"EleutherAI/gpt-neo-{size}"
        tokenizer = AutoTokenizer.from_pretrained(name)

        super().__init__(
            name=model,
            suffix=" ",
            device=device,
            maxlen=maxlen,
            model=AutoModelForCausalLM.from_pretrained(name).to(device).eval(),
            tokenizer=tokenizer,
        )

    @staticmethod
    def available_models():
        return [
            "gptneo.small",
            "gptneo.medium",
            "gptneo.large",
            "gptneo.xlarge",
        ]

    @staticmethod
    def default_maxlen():
        return 256
