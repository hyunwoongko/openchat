# OpenChat: Easy to use opensource chatting framework via neural networks
```
   ____   ____   ______ _   __   ______ __  __ ___   ______
  / __ \ / __ \ / ____// | / /  / ____// / / //   | /_  __/
 / / / // /_/ // __/  /  |/ /  / /    / /_/ // /| |  / /   
/ /_/ // ____// /___ / /|  /  / /___ / __  // ___ | / /    
\____//_/    /_____//_/ |_/   \____//_/ /_//_/  |_|/_/     
```
- OpenChat is easy to use opensource chatting framework.
- OpenChat supports 40+ dialogue model based on neural networks.
- You can talk with AI with **only one line of code.**

<br><br>

## Installation
```console
pip install openchat
```

<br><br>

## Supported Models
<ul>
<li>OpenChat supports 40+ dialogue models based on neural networks.</li>
<li>Use these names as parameter <code>model='name'</code> when you create <code>OpenChat</code>.</li>
<li><details>
  <summary>Click here if you want to check supported models.</summary>
  <h4><a href="https://github.com/EleutherAI/gpt-neo">GPT-Neo</a> </h4>
  <ul>
    <li>gptneo.small</li>
    <li>gptneo.medium</li>
    <li>gptneo.large</li>
    <li>gptneo.xlarge</li>
  </ul>
  <h4><a href="https://arxiv.org/abs/2004.13637">Blender</a></h4>
  <ul>
    <li>blender.small</li>
    <li>blender.medium</li>
    <li>blender.large</li>
    <li>blender.xlarge</li>
    <li>blender.xxlarge</li>
  </ul>
  <h4><a href="https://arxiv.org/abs/1911.00536">DialoGPT</a></h4>
  <ul>
    <li>dialogpt.small</li>
    <li>dialogpt.medium</li>
    <li>dialogpt.large</li>
  </ul>
  <h4><a href="https://arxiv.org/abs/1911.03768">Dodecathlon</a></h4>
  <ul>
    <li>dodecathlon.all_tasks_mt</li>
    <li>dodecathlon.convai2</li>
    <li>dodecathlon.wizard_of_wikipedia</li>
    <li>dodecathlon.empathetic_dialogues</li>
    <li>dodecathlon.eli5</li>
    <li>dodecathlon.reddit</li>
    <li>dodecathlon.twitter</li>
    <li>dodecathlon.ubuntu</li>
    <li>dodecathlon.image_chat</li>
    <li>dodecathlon.cornell_movie</li>
    <li>dodecathlon.light_dialog</li>
    <li>dodecathlon.daily_dialog</li>
  </ul>
  <h4><a href="https://arxiv.org/abs/2004.13637">Reddit</a></h4>
  <ul>
    <li>reddit.xlarge</li>
    <li>reddit.xxlarge</li>
  </ul>
  <h4><a href="https://arxiv.org/abs/2010.07079">Safety</a></h4>
  <ul>
    <li>safety.offensive</li>
    <li>safety.sensitive</li>
  </ul>
  <h4><a href="https://arxiv.org/abs/1911.03860">Unlikelihood</a></h4>
  <ul>
    <li>unlikelihood.wizard_of_wikipedia.context_and_label</li>
    <li>unlikelihood.wizard_of_wikipedia.context</li>
    <li>unlikelihood.wizard_of_wikipedia.label</li>
    <li>unlikelihood.convai2.context_and_label</li>
    <li>unlikelihood.convai2.context</li>
    <li>unlikelihood.convai2.label</li>
    <li>unlikelihood.convai2.vocab.alpha.1e-0</li>
    <li>unlikelihood.convai2.vocab.alpha.1e-1</li>
    <li>unlikelihood.convai2.vocab.alpha.1e-2</li>
    <li>unlikelihood.convai2.vocab.alpha.1e-3</li>
    <li>unlikelihood.eli5.context_and_label</li>
    <li>unlikelihood.eli5.context</li>
    <li>unlikelihood.eli5.label</li>
  </ul>
  <h4><a href="https://arxiv.org/abs/1811.01241">Wizard of Wikipedia</a></h4>
  <ul>
    <li>wizard_of_wikipedia.end2end_generator</li>
  </ul>
</details>
</li>
</ul>
<br><br>

## Usage
- Just import and create a object. That's all.
```python
>>> from openchat import OpenChat
>>> OpenChat(model="blender.medium", device="cpu")
```
<br><br>
   
- Set param `device='cuda'` If you want to use GPU acceleration.
```python
>>> from openchat import OpenChat
>>> OpenChat(model="blender.medium", device="cuda")
```
<br><br>

- Set param `device='cuda:n'` If you want to use a specific GPU.
```python
>>> from openchat import OpenChat
>>> OpenChat(model="blender.medium", device="cuda:2")  # <--- use 3rd GPU
>>> OpenChat(model="blender.medium", device="cuda:0")  # <--- use 1st GPU
```
<br><br>

- Set `**kwargs` if you want to change decoding options.
  - method (str): one of `["greedy", "beam", "top_k", "nucleus"]`,
  - num_beams (int): size of beam search 
  - top_k (int): K value for top-k sampling
  - top_p: (float): P value for nucleus sampling
  - no_repeat_ngram_size (int): beam search n-gram blocking size for removing repetition,
  - length_penalty (float): length penalty (1.0=None, UP=Longer, DOWN=Shorter)
- Decoding options must be `keyword argument` not `positional argument`.    
```python
>>> from openchat import OpenChat
>>> OpenChat(
...    model="blender.medium", 
...    device="cpu", 
...    method="top_k",
...    top_k=20,
...    no_repeat_ngram_size=3,
...    length_penalty=0.6,                            
... )
```

- For `safety.offensive` model, parameter `method` must be one of `["both", "string-match", "bert"]`
```python
>>> from openchat import OpenChat
>>> OpenChat(
...     model="safety.offensive",
...     device="cpu"
...     method="both" # ---> both, string-match, bert
... )

```
<br><br>

## Special Tasks
### 1. GPT-Neo
![](https://user-images.githubusercontent.com/38183241/113967262-972a8180-986b-11eb-9f02-68c9c093baf6.png)
- The GPT-Neo model was released in the EleutherAI/gpt-neo repository. 
- It is a GPT2 like causal language model trained on the Pile dataset.
- Openchat supports the above Prompt based dialogues via GPT-Neo.
- Below models provides custom prompt setting. (`*` means all models)
  - `gptneo.*`
<br><br>
  
### 2. ConvAI2
![](https://user-images.githubusercontent.com/38183241/112734380-bdf1d980-8f88-11eb-8ad7-18cf4d8d9ac6.png)
- ConvAI2 is one of the most famous conversational AI challenges about a persona. 
- Openchat provides custom persona setting like above image.
- Below models provides custom perona setting. (`*` means all models)
  - `blender.*`
  - `dodecathlon.convai2`
  - `unlikelihood.convai2.*`  
<br><br> 
    
### 3. Wizard of Wikipedia
![](https://user-images.githubusercontent.com/38183241/112734377-bb8f7f80-8f88-11eb-8c25-8c30691e29b8.png)
- Wizard of wikipedia is one of most famous knowledge grounded dialogue dataset.
- Openchat provides custom topic setting like above image.
- Below models provides custom topic setting. (`*` means all models)
    - `wizard_of_wikipedia.end2end_generator`
    - `dodecathlon.wizard_of_wikipedia`
    - `unlikelihood.wizard_of_wikipedia.*`
<br><br>

### 4. Safety Agents
![](https://user-images.githubusercontent.com/38183241/112735485-b41fa480-8f8f-11eb-9ac2-2c51a5294551.png)
![](https://user-images.githubusercontent.com/38183241/112735488-b71a9500-8f8f-11eb-94ce-55461c02966e.png)
- Openchat provides a dialog safety model to help you design conversation model.
- Below models provides dialog safety features.
  - `safety.offensive`: offensive words classification
  - `safety.sensitive`: sensitive topic classification

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
