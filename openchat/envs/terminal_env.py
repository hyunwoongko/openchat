from openchat.envs import BaseEnv
from openchat.models.base_model import BaseModel


class TerminalEnv(BaseEnv):
    """Dialogue for terminal environments"""

    def run(self, model: BaseModel):
        """Run model on terminal environments"""
        print(f"Chat with {model.name} !")

        while True:
            _in = input("user: ")

            if _in in self.keywords:
                # Format of self.keywords dictionary
                # self.keywords['/exit'] = (exit_function, 'good bye.')

                _out = self.keywords[_in][1]
                # text to print when keyword triggered

                print("bot: " + _out)
                self.keywords[_in][0]("USER_ID", _in)
                # function to operate when keyword triggered

            else:
                _out = model.predict("USER_ID", _in)
                print("bot: " + _out)
                # model inference and print message
