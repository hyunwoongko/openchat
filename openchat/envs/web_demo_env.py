import random
import base64

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from queue import Queue, Empty
from threading import Thread
import time
import traceback

from openchat.base.envs.base import BaseEnvironment
from openchat.base import (
    BaseAgent,
    ConvAI2Agent,
    WizardOfWikipediaAgent,
    SingleTurn,
    PromptAgent,
)


class WebServerEnvironment(BaseEnvironment):

    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 1
        self.CHECK_INTERVAL = 0.1
        self.app = Flask(__name__)
        self.requests_queue = Queue()
        self.users = []
        CORS(self.app)

    def start(self, agent: BaseAgent, **kwargs):
        ##
        # Request handler.
        # GPU app can process only one request in one time.
        def handle_requests_by_batch():
            while True:
                request_batch = []

                while not (len(request_batch) >= self.BATCH_SIZE):
                    try:
                        request_batch.append(self.requests_queue.get(timeout=self.CHECK_INTERVAL))
                    except Empty:
                        continue

                    for requests in request_batch:
                        try:
                            # 0 = user_id
                            # 1 = bot_id
                            # 2 = user_message
                            # 3 = topic
                            requests["output"] = generate(requests['input'][0],
                                                          requests['input'][1],
                                                          requests['input'][2],
                                                          requests['input'][3])
                        except Exception as e:
                            traceback.print_exc()
                            requests["output"] = e

        Thread(target=handle_requests_by_batch).start()

        # generate bot's message
        def generate(user_id, bot_id, user_message, topic):
            # add new user
            if user_id not in self.users:
                self.clear_histories(user_id)
                self.users.append(user_id)

            # max hold 10 user for memory
            if len(self.users) > 10:
                self.users.pop(0)

            if self.is_empty(user_id):
                self.pre_dialog_for_special_tasks(agent, user_id, bot_id, topic)

            if isinstance(agent, WizardOfWikipediaAgent):
                user_message = agent.retrieve_knowledge(user_message)

            if isinstance(agent, PromptAgent):
                user_message = f"{user_id}: {user_message}</s> <s>{bot_id}:"

            if isinstance(agent, SingleTurn):
                model_input = user_message
            else:
                model_input = self.make_model_input(
                    user_id,
                    user_message,
                    agent,
                )

            self.add_user_message(user_id, user_message)

            if isinstance(agent, PromptAgent):
                bot_message = agent.predict(
                    model_input,
                    person_1=user_id,
                    person_2=bot_id,
                    **kwargs,
                )["output"]

            else:
                bot_message = agent.predict(model_input, **kwargs)["output"]

            self.add_bot_message(user_id, bot_message)

            return bot_message

        ##
        # Sever health checking page.
        @self.app.route('/healthz', methods=["GET"])
        def health_check():
            return "Health", 200

        @self.app.route("/")
        def index():
            return render_template("index.html", title=agent.name.upper())

        @self.app.route('/send/<user_id>', methods=['POST'])
        def send(user_id):

            if self.requests_queue.qsize() > self.BATCH_SIZE:
                return jsonify({'message': 'Too Many Requests'}), 429

            try:
                text: str
                user_id = user_id.encode("UTF-8")
                user_id = base64.b64decode(user_id)
                user_id = user_id.decode("UTF-8")

                text = request.form['text']
                text = text.replace('<', '')
                text = text.replace('>', '')

                bot_id = request.form['bot_id']
                topic = request.form['topic']

            except Exception as e:
                return jsonify({'message': e}), 500

            try:
                if text == ".clear":
                    self.clear_histories(user_id)

                    return "Histories cleared."

                else:
                    args = [user_id, bot_id, text, topic]

                    # input a request on queue
                    req = {'input': args}
                    self.requests_queue.put(req)

                    # wait
                    while 'output' not in req:
                        time.sleep(self.CHECK_INTERVAL)

                    _out = req['output']

                return {"output": _out}

            except Exception as e:
                traceback.print_exc()
                return jsonify({'message': e}), 500

        from waitress import serve
        serve(self.app, port=80, host='0.0.0.0')

    def pre_dialog_for_special_tasks(self, agent, user_id, bot_id, topic):
        if isinstance(agent, ConvAI2Agent):
            return self.pre_dialog_for_convai2(agent, user_id)

        if isinstance(agent, WizardOfWikipediaAgent):
            return self.pre_dialog_for_wow(agent, topic)

        if isinstance(agent, PromptAgent):
            return self.pre_dialog_for_prompt(agent, user_id, bot_id, topic)

    def pre_dialog_for_prompt(self, agent, user_id, bot_id, topic):
        agent.name = bot_id

        story = f'{user_id} and {bot_id} are talking about {topic}.'
        story += f" {user_id} and {bot_id} start talking. "
        story += f"{user_id}: Hello {bot_id}. "
        story += f"{bot_id}: Hi {user_id}. "

        agent.add_prompt(
            self.histories,
            user_id,
            story,
        )

        return user_id, bot_id

    def pre_dialog_for_convai2(self, agent, user_id):
        _persona = f"[{agent.name.upper()}'s PERSONA]: "

        agent.add_persona(
            self.histories,
            user_id=user_id,
            text=_persona,
        )

    def pre_dialog_for_wow(self, agent, _topic):
        if _topic == ".topic":
            random_list = agent.topic_list
            random.shuffle(random_list)
            random_list = random_list[:4]

            _topic = f"[TOPIC]: {random_list}\n"

        else:
            if _topic in agent.topic_list:
                agent.set_topic(_topic)
            else:
                _topic = f"[TOPIC]: Wrong topic: {_topic}. Please enter validate topic.\n"
