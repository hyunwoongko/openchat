from transformers import GPT2LMHeadModel, GPT2Tokenizer
from openchat.agents import HuggingfaceAgent, DecoderLM


class DialoGPT(HuggingfaceAgent, DecoderLM):

    def __init__(self, model, device, maxlen=128):
        """
        DialoGPT was proposed in DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation
        by Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
        It’s a GPT2 Model trained on 147M conversation-like exchanges extracted from Reddit.

        Args:
            model (str): kinds of model
            device (str): computation device
            maxlen (int): max length for generation
        """

        model = self.check_model(model)
        name = f"microsoft/DialoGPT-{model.split('.')[-1]}"
        tokenizer = GPT2Tokenizer.from_pretrained(name)
        model = GPT2LMHeadModel.from_pretrained(name)

        super().__init__(
            name=model,
            prefix="",
            suffix="<|endoftext|>",
            device=device,
            maxlen=maxlen,
            model=model,
            tokenizer=tokenizer,
        )

    def available_models(self):
        return [
            "dialogpt.small",
            "dialogpt.medium",
            "dialogpt.large",
        ]