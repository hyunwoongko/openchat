from transformers import GPT2LMHeadModel, GPT2Tokenizer
from openchat.base import HuggingfaceAgent, DecoderLM


class DialoGPTAgent(HuggingfaceAgent, DecoderLM):

    def __init__(self, model, device, maxlen):
        model = self.check_agent(model)
        maxlen = maxlen if maxlen > 0 else self.default_maxlen()

        name = f"microsoft/DialoGPT-{model.split('.')[-1]}"
        tokenizer = GPT2Tokenizer.from_pretrained(name)

        super().__init__(
            name=model,
            suffix="<|endoftext|>",
            device=device,
            maxlen=maxlen,
            model=GPT2LMHeadModel.from_pretrained(name).to(device).eval(),
            tokenizer=tokenizer,
        )

    @staticmethod
    def available_models():
        return [
            "dialogpt.small",
            "dialogpt.medium",
            "dialogpt.large",
        ]

    @staticmethod
    def default_maxlen():
        return 48
