from openchat.base import ParlaiGenerationAgent


class ConvAI2Agent(ParlaiGenerationAgent):

    def add_persona(self, histories, user_id, text):
        histories[user_id]["prefix"].append(f"your persona: {text}")

    def clear_persona(self, histories, user_id):
        histories[user_id]["prefix"] = [
            pf for pf in histories[user_id]["prefix"]
            if "your persona:" not in pf
        ]
