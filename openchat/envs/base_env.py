import sys
from abc import abstractmethod, ABC
from openchat.models.base_model import BaseModel


class BaseEnv(ABC):
    """
    Base dialogue environment class
    """

    def __init__(self):
        self.histories = {
            # 1. Format of dialogue histories
            #   'USER_ID': {'user': [UTTERANCES BY USER], 'bot': [UTTERANCES BY BOT]}
            #
            # 2. Examples of dialogue histories
            #   'user_1': {'user': ['Hi.', 'What is your name?'] , 'bot': ['Hello.', 'My name is Meena!']},
            #   'user_2': {'user': ['Hello.', 'What is your name?'] , 'bot': ['Hi.', 'My name is Blender!']},
        }

        self.keywords = {
            ".exit": (self.exit, "good bye."),
            ".clear": (self.clear, "histories cleared."),
            # .keyword: (function, message)
        }

    @abstractmethod
    def run(self, model: BaseModel):
        """Start to dialogue"""

        return NotImplemented

    @staticmethod
    def exit(user_id, text):
        exit(0)
        sys.exit(0)

    def clear(self, user_id: str, text: str) -> None:
        """
        clear all dialogue histories

        Args:
            user_id (str): user id to clear histories
            text (str): input text from user
        """

        self.histories[user_id] = {"user": [], "bot": []}

    def add_keyword(
        self,
        keyword: str,
        message: str,
        func,
    ) -> None:
        """
        Add new keywords with user-defined function
        default keywords dictionary has two keywords: '/exit', '/clear'

        Args:
            keyword (str): keyword to trigger function
            message (str): text to print when keyword triggered
            func (function): function to operate when keyword triggered
        """

        self.keywords[keyword] = (func, message)
