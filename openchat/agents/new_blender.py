from transformers import (
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)
from openchat.base import HuggingfaceAgent

class NewBlenderAgent(HuggingfaceAgent):

    def __init__(self, model, device, maxlen):
        model = self.check_agent(model)
        maxlen = maxlen if maxlen > 0 else self.default_maxlen()

        if "xlarge" in model:
            size = "-3B"
        elif "large" in model:
            size = "-1B-distill"
        elif "medium" in model:
            size = "-400M-distill"
        elif "small" in model:
            size = "_small-90M"
        else:
            raise Exception("wrong model")
        
        name = f"facebook/blenderbot{size}"
        
        if size == "_small-90M":
            gen_model = BlenderbotSmallForConditionalGeneration.from_pretrained(name)
            tokenizer = BlenderbotSmallTokenizer.from_pretrained(name)
        else:
            gen_model = BlenderbotForConditionalGeneration.from_pretrained(name)
            tokenizer = BlenderbotTokenizer.from_pretrained(name)

        super().__init__(
            name=model,
            suffix="<|endoftext|>",
            device=device,
            maxlen=maxlen,
            model=gen_model.to(device).eval(),
            tokenizer=tokenizer,
        )

    @staticmethod
    def available_models():
        return [
            "blender.small",
            "blender.medium",
            "blender.large",
            "blender.xlarge",
        ]

    @staticmethod
    def default_maxlen():
        return 48
