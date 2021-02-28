# DialoGPT chatting
- package for chatting with DialoGPT.
- you can check details of model here: [arXiv](https://arxiv.org/abs/1911.00536).
<br><br>

## Installation
```console
pip install dialogpt-chat
```
<br><br>

## Usage
### 1. Terminal mode
- chatting with DialoGPT on terminal.
- The model size must be one of the [small, medium, large].
- type '/exit' if you want to exit dialogue.

```python
>>> from dialobot_chat import DialoGPT
>>> gpt = DialoGPT(size="large", device="cuda")
>>> gpt.run()
```
```
user : Hello.
bot : How are you?
user : I'm good. how about you?
bot : Good, you?
user : Me too.
bot : That's good.
user : /exit
bot : bye.
```
<br>

### 2. Chatting by user id.
- chatting with DialoGPT by user id. (single-turn)
```python
>>> from dialobot_chat import DialoGPT
>>> gpt = DialoGPT(size="large", device="cuda")
>>> gpt.predict(user_id="USER_ID", text="Hello.")
```
```
How are you?
```
<br>

- chatting with DialoGPT by user id. (multi-turn)
```python
>>> from dialobot_chat import DialoGPT
>>> gpt = DialoGPT(size="large", device="cuda")

>>> while True:
...    _in = input('user : ')
...    _out = gpt.predict(user_id="USER_ID", text=_in)
...    print(f"bot : {_out}")
```
```
user : Hello.
bot : How are you?
user : I'm good. how about you?
bot : Good, you?
user : Me too.
bot : That's good.
user : /exit
bot : bye.
```
<br>

### 3. Checking dialogue histories
```python
>>> from dialobot_chat import DialoGPT
>>> gpt = DialoGPT(size="large", device="cuda")
>>> gpt.histories
```
```
{
    user_1 : {'user': [] , 'bot': []},
    user_2 : {'user': [] , 'bot': []},
    ...more...
    user_n : {'user': [] , 'bot': []},
}
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