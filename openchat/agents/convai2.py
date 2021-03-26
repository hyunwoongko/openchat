from openchat.agents import ParlaiGenerationAgent


class ConvAI2Agent(ParlaiGenerationAgent):

    def add_persona(self, persona: str):
        self.prefix += f"your persona: {persona}\n"

    def clear_persona(self):
        self.prefix = ""
