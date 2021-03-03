from typing import Dict
from flask import Flask, render_template
from flask_cors import CORS

from openchat.envs import BaseEnv
from openchat.models import BaseModel


class WebDemoEnv(BaseEnv):

    def __init__(self):
        super().__init__()
        self.app = Flask(__name__)
        CORS(self.app)

    def run(self, model: BaseModel):

        @self.app.route("/")
        def index():
            return render_template("index.html", title=model.name)

        @self.app.route('/send/<user_id>/<text>', methods=['GET'])
        def send(user_id, text: str) -> Dict[str, str]:

            if text in self.keywords:
                # Format of self.keywords dictionary
                # self.keywords['/exit'] = (exit_function, 'good bye.')

                _out = self.keywords[text][1]
                # text to print when keyword triggered

                self.keywords[text][0](user_id, text)
                # function to operate when keyword triggered

            else:
                _out = model.predict(user_id, text)

            return {"output": _out}

        self.app.run(host="0.0.0.0", port=8080)
