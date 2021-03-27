import random
import sys

from openchat.base.agents.base import SingleTurn
from openchat.base.envs.base import BaseEnvironment
from openchat.base import (
    BaseAgent,
    ConvAI2Agent,
    WizardOfWikipediaAgent,
)

from openchat.utils.terminal_utils import (
    cprint,
    cinput,
    Colors,
)


class TerminalEnvironment(BaseEnvironment):

    def __init__(
        self,
        user_color=Colors.GREEN,
        bot_color=Colors.YELLOW,
        special_color=Colors.BLUE,
        system_color=Colors.MAGENTA,
    ):
        super().__init__()
        self.user_id = "dummy_value"
        self.user_color = user_color
        self.bot_color = bot_color
        self.special_color = special_color
        self.system_color = system_color

    def start(self, agent: BaseAgent, **kwargs):
        cprint(
            f"\n[SYSTEM]: Let's talk with [{agent.name.upper()}].\n"
            f"[SYSTEM]: Enter '.exit', if you want to exit chatting.\n"
            f"[SYSTEM]: Enter '.reset', if you want reset all histories.\n",
            color=self.system_color)

        self.clear_histories(self.user_id)

        while True:
            if self.is_empty(self.user_id):
                self.pre_dialog_for_special_tasks(agent)

            user_message = cinput("[USER]: ", color=self.user_color)

            if user_message == ".exit":
                cprint(
                    f"[SYSTEM]: good bye.\n",
                    color=self.system_color,
                )
                exit(0)
                sys.exit(0)

            if user_message == ".reset":
                cprint(
                    f"[SYSTEM]: reset all histories.\n",
                    color=self.system_color,
                )
                self.clear_histories(self.user_id)
                continue

            if isinstance(agent, WizardOfWikipediaAgent):
                user_message = agent.retrieve_knowledge(user_message)

            if isinstance(agent, SingleTurn):
                model_input = user_message
            else:
                model_input = self.make_model_input(
                    self.user_id,
                    user_message,
                    agent,
                )

            self.add_user_message(self.user_id, user_message)
            bot_message = agent.predict(model_input, **kwargs)["output"]

            self.add_bot_message(self.user_id, bot_message)
            cprint(
                f"[{agent.name.upper()}]: {bot_message}",
                color=self.bot_color,
            )

    def pre_dialog_for_special_tasks(self, agent):
        if isinstance(agent, ConvAI2Agent):
            self.pre_dialog_for_convai2(agent)

        if isinstance(agent, WizardOfWikipediaAgent):
            self.pre_dialog_for_wow(agent)

    def pre_dialog_for_convai2(self, agent):
        cprint(
            f"[SYSTEM]: Please input [{agent.name.upper()}]'s perosna.\n"
            f"[SYSTEM]: Enter '.done' if you want to end input persona.\n",
            color=Colors.MAGENTA)

        while True:
            _persona = cinput(
                f"[{agent.name.upper()}'s PERSONA]: ",
                color=self.special_color,
            )

            if _persona == ".done":
                cprint(
                    f"[{agent.name.upper()}'s PERSONA]: Persona setting complete.\n",
                    color=self.special_color,
                )
                break
            else:
                agent.add_persona(
                    self.histories,
                    user_id=self.user_id,
                    text=_persona,
                )

    def pre_dialog_for_wow(self, agent):
        cprint(
            f"[SYSTEM]: Please input topic for Wizard of wikipedia.\n"
            f"[SYSTEM]: Enter '.topic' if you want to check random topic examples.\n",
            color=Colors.MAGENTA)

        while True:
            _topic = cinput(
                "[TOPIC]: ",
                color=self.special_color,
            )

            if _topic == ".topic":
                random_list = agent.topic_list
                random.shuffle(random_list)
                random_list = random_list[:4]

                _topic = cprint(
                    f"[TOPIC]: {random_list}\n",
                    color=self.special_color,
                )

            else:
                if _topic in agent.topic_list:
                    cprint(
                        f"[TOPIC]: Topic setting complete.\n",
                        color=self.special_color,
                    )
                    agent.set_topic(_topic)
                    break
                else:
                    _topic = cprint(
                        f"[TOPIC]: Wrong topic: {_topic}. Please enter validate topic.\n",
                        color=self.special_color,
                    )
