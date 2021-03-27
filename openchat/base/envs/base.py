from abc import ABC, abstractmethod
from dataclasses import dataclass
from openchat.base import BaseAgent, DecoderLM


@dataclass
class Command:
    command: str
    description: str
    function: callable
    message: str

    def __str__(self):
        return f"command '{self.command}': {self.description}"

    def __repr__(self):
        return f"command '{self.command}': {self.description}"


class BaseEnvironment(ABC):

    def __init__(self):
        clear_command = Command(
            command=".reset",
            description="reset all histories.",
            function=self.clear_histories,
            message="all histories are moved.",
        )

        self.commands = {".clear": clear_command}
        self.histories = {}

    def clear_histories(self, user_id, text=None):
        self.histories[user_id] = {
            "user_message": [],
            "bot_message": [],
            "model_input": "",
            "prefix": [],
            "chosen_topic": ","
        }

    def help_message(self):
        return "\n".join([str(_) for _ in self.commands])

    def add_command(self, command: Command):
        self.commands[command.command] = command

    def add_user_message(self, user_id, text):
        self.histories[user_id]["user_message"].append(text)

    def add_bot_message(self, user_id, text):
        self.histories[user_id]["bot_message"].append(text)

    def make_model_input(self, user_id, user_input, agent):
        prefix = self.histories[user_id]["prefix"]

        if len(prefix) > 0:
            prefix = agent.suffix.join(prefix) + agent.suffix
        else:
            prefix = ""

        if isinstance(agent, DecoderLM):
            user_input += agent.suffix

        current_tokens = agent.tokenizer(prefix + user_input)["input_ids"]
        histories_for_current_turn = []
        num_history_tokens = len(current_tokens)

        for u, m in zip(
                reversed(self.histories[user_id]["user_message"]),
                reversed(self.histories[user_id]["bot_message"]),
        ):

            history = u + agent.suffix + m + agent.suffix
            tokens = agent.tokenizer(history)["input_ids"]

            num_history_tokens += len(tokens)
            if num_history_tokens < agent.maxlen:
                histories_for_current_turn.append(history)
            else:
                break

        histories_for_current_turn = list(reversed(histories_for_current_turn))
        return prefix + "".join(histories_for_current_turn) + user_input

    @abstractmethod
    def start(self, agent: BaseAgent):
        raise NotImplemented
