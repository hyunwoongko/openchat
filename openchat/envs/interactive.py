import random
import sys

from openchat.base.agents.base import SingleTurn
from openchat.base.agents.prompt import PromptAgent
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


class InteractiveEnvironment(BaseEnvironment):

    def __init__(
        self,
        user_color=Colors.GREEN,
        bot_color=Colors.YELLOW,
        special_color=Colors.BLUE,
        system_color=Colors.CYAN,
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
                pre_dialog_output = self.pre_dialog_for_special_tasks(agent)

            if isinstance(agent, PromptAgent):
                user_name, bot_name = pre_dialog_output
                user_message = cinput(f"[{user_name.upper()}]: ",
                                      color=self.user_color)
            else:
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

            if isinstance(agent, PromptAgent):
                user_message = f"{user_name}: {user_message} {bot_name}:"

            if isinstance(agent, SingleTurn):
                model_input = user_message
            else:
                model_input = self.make_model_input(
                    self.user_id,
                    user_message,
                    agent,
                )

            self.add_user_message(self.user_id, user_message)

            if isinstance(agent, PromptAgent):
                bot_message = agent.predict(
                    model_input,
                    person_1=user_name,
                    person_2=bot_name,
                    **kwargs,
                )["output"]

            else:
                bot_message = agent.predict(model_input, **kwargs)["output"]

            cprint(
                f"[{agent.name.upper()}]: {bot_message}",
                color=self.bot_color,
            )

            if isinstance(agent, PromptAgent):
                self.add_bot_message(self.user_id, bot_message)
            else:
                self.add_bot_message(self.user_id,
                                     f"{bot_name}: {bot_message} ")

    def pre_dialog_for_special_tasks(self, agent):
        if isinstance(agent, ConvAI2Agent):
            return self.pre_dialog_for_convai2(agent)

        if isinstance(agent, WizardOfWikipediaAgent):
            return self.pre_dialog_for_wow(agent)

        if isinstance(agent, PromptAgent):
            return self.pre_dialog_for_prompt(agent)

    def pre_dialog_for_prompt(self, agent):
        user_name = cinput(
            f"[YOUR NAME]: ",
            color=self.special_color,
        )

        bot_name = cinput(
            f"[{agent.name.upper()}'s NAME]: ",
            color=self.special_color,
        )

        agent.name = bot_name

        cprint(
            f"\n[SYSTEM]: Please input story you want.\n"
            f"[SYSTEM]: The story must contains '{user_name}' and '{bot_name}'.\n",
            color=self.system_color)

        story = cinput(
            "[STORY]: ",
            color=self.special_color,
        )

        while (user_name not in story) or (bot_name not in story):
            cprint(
                f"\n[SYSTEM]: Please input story you want.\n"
                f"[SYSTEM]: The story MUST contains '{user_name}' and '{bot_name}'.\n",
                color=self.system_color)

            story = cinput(
                "[STORY]: ",
                color=self.special_color,
            )

        cprint(
            f"[STORY]: Story setting complete.\n",
            color=self.special_color,
        )

        story += f" {user_name} and {bot_name} start talking. "
        story += f"{user_name}: Hello {bot_name}. "
        story += f"{bot_name}: Hi {user_name}. "

        agent.add_prompt(
            self.histories,
            self.user_id,
            story,
        )

        return user_name, bot_name

    def pre_dialog_for_convai2(self, agent):
        cprint(
            f"[SYSTEM]: Please input [{agent.name.upper()}]'s perosna.\n"
            f"[SYSTEM]: Enter '.done' if you want to end input persona.\n",
            color=self.system_color)

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
            color=self.system_color)

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
