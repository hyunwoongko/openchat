from openchat import OpenChat
from demo.web_demo_env import WebDemoEnv

OpenChat(model="blenderbot", size="large", env=WebDemoEnv())