from parlai.core.agents import create_agent_from_model_file
from openchat.agents import ConvAI2Agent, Seq2SeqLM


class Blender(ConvAI2Agent, Seq2SeqLM):

    def __init__(
        self,
        model: str,
        device: str,
        maxlen: int = 256,
    ) -> None:
        """
        The Blender chatbot model was proposed in Recipes for building an open-domain chatbot
        Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.
        """

        model = self.check_model(model)

        if "small" in model:
            size = "90M"
        elif "medium" in model:
            size = "400Mdistill"
        elif "large" in model:
            size = "1Bdistill"
        elif "xlarge" in model:
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
            model=create_agent_from_model_file(
                f"zoo:blender/blender_{size}/model"),
        )

    def available_models(self):
        return [
            "blender.small",
            "blender.medium",
            "blender.large",
            "blender.xlarge",
            "blender.xxlarge",
        ]
