from parlai.core.agents import create_agent_from_model_file
from openchat.agents import ParlaiGenerationAgent, Seq2SeqLM


class Reddit(ParlaiGenerationAgent, Seq2SeqLM):

    def __init__(
        self,
        model: str,
        device: str,
        maxlen: int = 256,
    ) -> None:
        self.check_model(model)

        if "xlarge" in model:
            size = "3B"
        elif "xxlarge" in model:
            size = "9B"
        else:
            raise Exception("wrong model")

        super().__init__(
            name=model,
            prefix="",
            suffix="\n",
            device=device,
            maxlen=maxlen,
            model=create_agent_from_model_file(f"zoo:blender/reddit_{size}/model"),
        )

    def available_models(self):
        return [
            "reddit.xlarge",
            "reddit.xxlarge",
        ]