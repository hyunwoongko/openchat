import random
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
    ):
        super().__init__()
        self.user_id = "dummy_value"
        self.user_color = user_color
        self.bot_color = bot_color
        self.special_color = special_color

    def start(self, agent: BaseAgent, **kwargs):
        cprint(
            f"\n[SYSTEM]: Let's talk with [{agent.name.upper()}].\n"
            f"[SYSTEM]: Enter '.help', if you want to see chatting commands.\n",
            color=Colors.MAGENTA)

        self.clear_histories(self.user_id, text=None)
        self.pre_dialog_for_special_tasks(agent)

        while True:
            user_message = cinput("[USER]: ", color=self.user_color)
            if isinstance(agent, WizardOfWikipediaAgent):
                user_message = agent.retrieve_knowledge(user_message)

            model_input = self.make_model_input(
                self.user_id,
                user_message,
                agent,
            )

            self.add_user_message(self.user_id, user_message)
            bot_message = agent.predict(model_input)["output"]

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
                    f"[TOPIC]: Random examples = {random_list}\n",
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
