from parlai.core.agents import create_agent_from_model_file
from openchat.base import ParlaiGenerationAgent, Seq2SeqLM


class RedditAgent(ParlaiGenerationAgent, Seq2SeqLM):

    def __init__(self, model: str, device: str, maxlen) -> None:
        self.check_agent(model)
        maxlen = maxlen if maxlen > 0 else self.default_maxlen()

        if "xlarge" in model:
            size = "3B"
        elif "xxlarge" in model:
            size = "9B"
        else:
            raise Exception("wrong model")

        super().__init__(
            name=model,
            suffix="\n",
            device=device,
            maxlen=maxlen,
            model=create_agent_from_model_file(
                f"zoo:blender/reddit_{size}/model"),
        )

    @staticmethod
    def available_models():
        return [
            "reddit.xlarge",
            "reddit.xxlarge",
        ]

    @staticmethod
    def default_maxlen():
        return 256
