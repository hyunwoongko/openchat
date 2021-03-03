from typing import Union

from openchat.envs import BaseEnv, TerminalEnv
from openchat.models.blenderbot import BlenderBot
from openchat.models.dialogpt import DialoGPT


class OpenChat(object):

    def __init__(
        self,
        model: str,
        size: str,
        device: str = "cpu",
        env: Union[BaseEnv, str] = TerminalEnv(),
        max_context_length=128,
    ) -> None:
        """
        Constructor for OpenChat

        Args:
            env (Union[BaseEnv, str]): dialogue environment
            model (str): generative dialogue model
            size (str): model size (It may vary depending on the model)
            device (str): device argument
            max_context_length (int): max history context length
                (it means that length of input context tokens)
        """

        print("""
           ____   ____   ______ _   __   ______ __  __ ___   ______
          / __ \ / __ \ / ____// | / /  / ____// / / //   | /_  __/
         / / / // /_/ // __/  /  |/ /  / /    / /_/ // /| |  / /   
        / /_/ // ____// /___ / /|  /  / /___ / __  // ___ | / /    
        \____//_/    /_____//_/ |_/   \____//_/ /_//_/  |_|/_/     
                        
                             ... LOADING ...
        """)

        self.size = size
        self.device = device
        self.max_context_length = max_context_length
        self.env = env

        self.model = self.select_model(model)
        self.model.run()

    def select_model(self, model):
        assert model in self.available_models(), \
            f"Unsupported model. available models: {self.available_models()}"

        if model == "dialogpt":
            return DialoGPT(
                size=self.size,
                env=self.env,
                max_context_length=self.max_context_length,
                device=self.device,
            )

        elif model == "blenderbot":
            return BlenderBot(
                size=self.size,
                env=self.env,
                max_context_length=self.max_context_length,
                device=self.device,
            )

    def available_models(self):
        return ["dialogpt", "blenderbot"]
