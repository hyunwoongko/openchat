# OpenChat: Opensource chatting framework for generative models
```
    ____   ____   ______ _   __   ______ __  __ ___   ______
   / __ \ / __ \ / ____// | / /  / ____// / / //   | /_  __/
  / / / // /_/ // __/  /  |/ /  / /    / /_/ // /| |  / /   
 / /_/ // ____// /___ / /|  /  / /___ / __  // ___ | / /    
 \____//_/    /_____//_/ |_/   \____//_/ /_//_/  |_|/_/     
```
- OpenChat is opensource chatting framework for generative models.
- You can talk with AI with **only one line of code.**

<br><br>

## Installation
```console
pip install openchat
```

<br><br>

## Model Supports
- [DialoGPT](https://arxiv.org/abs/1911.00536) : [small, medium, large]
- [BlenderBot](https://arxiv.org/abs/2004.13637) : [small, medium, large, xlarge]
- Coming Soon...

<br><br>

## Usage
### 1. Terminal environment
- Chatting on terminal.
- Type `.exit` if you want to exit dialogue.
- Type `.clear` if you want to clear all histories.
- You can add additional user keywords. (Details are described below.)

```python
>>> from openchat import OpenChat
>>> OpenChat(model="blenderbot", size="large")
```
```
Chat with facebook/blenderbot-1B-distill !
user : Hi. It is nice day.
bot : It is a nice day out. I am going to go for a walk. What are you up to?
user : I'll play computer game. Do you like playing game?
bot : I love playing video games. What kind of games do you like to play on your computer?
user : I love racing game. It is so exited. How about you?
bot : I love racing games as well. I like to play on the Playstation 4. What about you?
user : Usually I like to play on PC. I have a question. Who is CEO of Apple?
bot : The CEO of Apple is Steve Jobs. He founded the company in 1976.
user : That's right. I have an IPhone. It is my favorite phone. What is your phone?
bot : I have a Samsum Galaxy S8. I love it. It's the best phone I've ever had.
user : .exit
bot : good bye.
```
<br>

- If you want to use GPU, use argument `device="cuda"`
```python
>>> from openchat import OpenChat
>>> OpenChat(model="blenderbot", size="large", device="cuda")
```

<br>

### 2. Your own environment (not terminal)
- You can use user id to manage user-specific history.
- This can be useful when deployed on Facebook messenger or WhatsApp.
- There is a web demo implementation in the `/demo` folder.

![](https://user-images.githubusercontent.com/38183241/109745263-f1a24380-7c16-11eb-8ea2-915b2ed281ab.png)

<br>
    
#### 2.1. Write your own environment class
- Make your own environment class inherited from `BaseEnv`
- And implement your own `run(model: BaseModel)` method like below.

```python
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
```
<br>

#### 2.2. Start to run application.
```python
from openchat import OpenChat
from demo.web_demo_env import WebDemoEnv

OpenChat(model="blenderbot", size="large", env=WebDemoEnv())
```
<br><br>

### 3. Additional Options
#### 3.1. Add custom Keywords

- You can add new manual keyword such as `.exit`, `.clear`, 
- call the `self.add_keyword('.new_keyword', 'message to print', triggered_function)'` method.
- `triggered_function` should be form of `function(user_id:str, text:str)`

```python
from openchat.envs import BaseEnv


class YourOwnEnv(BaseEnv):
    
    def __init__(self):
        super().__init__()
        self.add_keyword(".new_keyword", "message to print", self.function)

    def function(self, user_id: str, text: str):
        """do something !"""
        
```
<br><br>

#### 3.2. Modify generation options
- You can modify max_context_length (number of input history tokens, default is 128).
```python
>>> OpenChat(size="large", device="cuda", max_context_length=256)
```
<br>

- You can modify generation options `['num_beams', 'top_k', 'top_p']`.
```python
>>> model.predict(
...     user_id="USER_ID",
...     text="Hello.",
...     num_beams=5,
...     top_k=20,
...     top_p=0.8,
... )
```
<br><br>

#### 3.3. Check histories
- You can check all dialogue history using `self.histories`
```python
from openchat.envs import BaseEnv


class YourOwnEnv(BaseEnv):
    
    def __init__(self):
        super().__init__()
        print(self.histories)
```
```
{
    user_1 : {'user': [] , 'bot': []},
    user_2 : {'user': [] , 'bot': []},
    ...more...
    user_n : {'user': [] , 'bot': []},
}
```
<br>

#### 3.4. Clear histories
- You can clear all dialogue histories
```python
from flask import Flask
from openchat.envs import BaseEnv
from openchat.models import BaseModel

class YourOwnEnv(BaseEnv):
    
    def __init__(self):
        super().__init__()
        self.app = Flask(__name__)

    def run(self, model: BaseModel):
        
        @self.app.route('/send/<user_id>/<text>', methods=['GET'])
        def send(user_id, text: str) -> Dict[str, str]:
            
            self.clear(user_id, text)
            # clear all histories ! 

```

<br><br>

## License
```
Copyright 2021 Hyunwoong Ko.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
