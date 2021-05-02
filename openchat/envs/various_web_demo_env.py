import random
import base64
from collections import OrderedDict
import torch

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from queue import Queue, Empty
from threading import Thread
import time
import traceback
import re
import gc
from openchat.base.envs.base import BaseEnvironment
from openchat.base import (
    BaseAgent,
    ConvAI2Agent,
    WizardOfWikipediaAgent,
    SingleTurn,
    PromptAgent,
)

class VariousWebServerEnvironment(BaseEnvironment):

    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 1
        self.CHECK_INTERVAL = 0.1
        self.app = Flask(__name__)
        self.requests_queue = Queue()
        self.users = OrderedDict()
        self.agents = {}    # key=agent name, value=agent obj
        self.max_hold_user = 50
        CORS(self.app)

    def start(self, agents: list, **kwargs):
        remove_token = re.compile('</?name[03456789]>', re.I)

        # parsing conformed model name and obj
        for agent_obj in agents:
            agent_obj: BaseAgent
            self.agents[agent_obj.name.upper()] = agent_obj

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
                            # 4 = agent
                            requests["output"] = generate(requests['input'][0],
                                                          requests['input'][1],
                                                          requests['input'][2],
                                                          requests['input'][3],
                                                          requests['input'][4],)
                        except Exception as e:
                            traceback.print_exc()
                            requests["output"] = e

        Thread(target=handle_requests_by_batch).start()

        # generate bot's message
        def generate(user_id, bot_id, user_message, topic, agent: str):
            gc.enable()
            torch.cuda.empty_cache()
            try:
                # add new user
                if user_id not in self.users:
                    self.clear_histories(user_id)
                    self.users[user_id] = [topic, agent]

                # get agent obj
                agent = agent.upper()
                try:
                    agent_obj = self.agents[agent]
                except:
                    return "Wrong agent!"

                # max hold 50 user for memory
                if len(self.users) > self.max_hold_user:
                    old_user = self.users.popitem(last=False)[0]
                    self.remove_user_in_histories(old_user)

                if self.is_empty(user_id):
                    self.pre_dialog_for_special_tasks(agent_obj, user_id, bot_id, topic)

                # When the agent or topic is changed, init again
                if topic != self.users[user_id][0] or agent != self.users[user_id][1]:
                    self.histories[user_id]["prefix"] = []
                    self.clear_histories(user_id)
                    self.users[user_id] = [topic, agent]
                    self.pre_dialog_for_special_tasks(agent_obj, user_id, bot_id, topic)

                if isinstance(agent_obj, WizardOfWikipediaAgent):
                    user_message = agent_obj.retrieve_knowledge(user_message)

                if isinstance(agent_obj, PromptAgent):
                    #user_message.replace(user_id, '<name1>').replace(bot_id, '<name2>')
                    user_message = user_id + f" : {user_message} " + bot_id + " :"

                if isinstance(agent_obj, SingleTurn):
                    model_input = user_message
                else:
                    model_input = self.make_model_input(
                        user_id,
                        user_message,
                        agent_obj,
                    )

                self.add_user_message(user_id, user_message)

                if isinstance(agent_obj, PromptAgent):
                    bot_message = agent_obj.predict(
                        model_input,
                        person_1=user_id,
                        person_2=bot_id,
                        **kwargs,
                    )["output"]

                    bot_message = remove_token.sub("", bot_message)

                    self.add_bot_message(user_id, bot_message)

                    #bot_message = bot_message.replace('<name1>', user_id).replace('<name2>', bot_id)
                    #bot_message = bot_message.replace('<', '').replace('>', '')
                else:
                    bot_message = agent_obj.predict(model_input, **kwargs)["output"]
                    if "A:" in bot_message :
                        bot_message = "I'm so sorry I didn't quite get that, can you repeat it?"
                    if bot_message == "A" or bot_message == "A ":
                        bot_message = "I'm so sorry I didn't quite get that, can you repeat it?"
                    self.add_bot_message(user_id, bot_message)
                
                bot_messages = bot_message.split(user_id+" :")
                bot_messages = bot_messages[0].split(":")

                return bot_messages[0]

            except:
                traceback.print_exc()

                return "Error :("
            finally:
                gc.collect()

        ##
        # Sever health checking page.
        @self.app.route('/healthz', methods=["GET"])
        def health_check():
            return "Health", 200

        @self.app.route("/")
        def index():
            return render_template("index_for_various.html", titles=list(self.agents.keys()))

        # Get base64 encoded user id
        @self.app.route("/base64", methods=['POST'])
        def get_base64():
            try:
                user_id = request.form['user_id']
                user_id = user_id.encode("UTF-8")
                user_id = base64.b64encode(user_id).decode("UTF-8")

                return user_id
            except:
                return "error"

        @self.app.route('/send/<user_id>', methods=['POST'])
        def send(user_id):

            if self.requests_queue.qsize() > self.BATCH_SIZE:
                return {'output': 'Too Many Requests'}, 429

            try:
                text: str

                try:
                    user_id_64 = user_id.encode("UTF-8")
                    user_id_64 = base64.b64decode(user_id_64)
                    user_id = user_id_64.decode("UTF-8")
                except:
                    user_id = user_id

                text = request.form['text']
                text = text.replace('<', '"')
                text = text.replace('>', '"')

                bot_id = "OpenChatAI" # request.form['bot_id'].replace('<', '"').replace('>', '"')
                topic = "I am OpenChatAI and I am chatting with my friend " + user_id + " on the internet using an instant messaging application.  We are enjoying talking to each other. Our conversation begins here."# request.form['topic'].replace('<', '"').replace('>', '"')
                agent = request.form['agent']   # agent's name

            except Exception as e:
                return {"output": 'Bad request'}, 500

            try:
                if text == ".clear":
                    self.clear_histories(user_id)

                    return {"output": "Histories cleared."}

                elif text == ".exit":
                    self.remove_user_in_histories(user_id)
                    del self.users[user_id]

                    return {"output": "Goodbye, friend."}

                else:
                    args = [user_id, bot_id, text, topic, agent]

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
                return {'output': 'Sorry, there was an error.'}, 500


        from waitress import serve
        #serve(self.app, host='0.0.0.0', port=80)
        #serve(self.app, host='0.0.0.0', port=8000)
        self.app.run(host='0.0.0.0', port=8000)


    def pre_dialog_for_special_tasks(self, agent, user_id, bot_id, topic):
        if isinstance(agent, ConvAI2Agent):
            return self.pre_dialog_for_convai2(agent, user_id)

        if isinstance(agent, WizardOfWikipediaAgent):
            return self.pre_dialog_for_wow(agent, topic)

        if isinstance(agent, PromptAgent):
            return self.pre_dialog_for_prompt(agent, user_id, bot_id, topic)

    def pre_dialog_for_prompt(self, agent, user_id, bot_id, topic):
        agent.name = bot_id


        story = topic
        
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
