# DialoGPT Chat
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
- The model size must be one of the `['small', 'medium', 'large']`.
- type `/exit` if you want to exit dialogue.
- type `/clear` if you want to clear all histories

```python
>>> from dialobot_chat import DialoGPT
>>> gpt = DialoGPT(size="large", device="cuda")
>>> gpt.run()
```
```
user : Hello.
bot : How are you?
user : I'm great. it is a nice day.
bot : That's good.
user : Who is CEO of Apple?
bot : Steve Jobs.
user : /clear
bot : history cleared.
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
user : I'm great. it is a nice day.
bot : That's good.
user : Who is CEO of Apple?
bot : Steve Jobs.
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
<br>

### 4. Clear dialogue histories
```python
>>> from dialobot_chat import DialoGPT
>>> gpt = DialoGPT(size="large", device="cuda")
>>> gpt.clear(user_id="USER_ID")
```
<br>

### 5. Additional options
- you can modify max_context_length (number of input history tokens, default is 48).
```python
>>> gpt = DialoGPT(size="large", device="cuda", max_context_length=96)
```
<br>

- you can modify generation options `['num_beams', 'top_k', 'top_p']`.
```python
>>> gpt.predict(
...     user_id="USER_ID",
...     text="Hello.",
...     num_beams=5,
...     top_k=20,
...     top_p=0.8,
... )
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
