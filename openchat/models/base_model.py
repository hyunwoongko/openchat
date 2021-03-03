from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, name: str, env):
        """
        Args:
            name (str): model name
            env (BaseEnv): dialogue manager
        """
        self.name = name
        self.env = env

    @abstractmethod
    def predict(self, user_id: str, text: str) -> str:
        """
        Predict output from histories and input text

        Args:
            user_id (str): user's ID
            text (str): user's input text
        """

        return NotImplemented

    def run(self):
        self.env.run(self)
