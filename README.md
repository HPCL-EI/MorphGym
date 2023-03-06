# MorphGym

A training platform for Morphology Intelligence


## Installation

1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
   - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)

4. Install RL-Games
   * https://github.com/Denys88/rl_games



## Use preinstalled worlds

```python
import morphgym as mg
agent,env = mg.make('Dog-2D','Plane',config=None) # agent name, env name, config path (optional)

for _ in range(50): # morph searching
    morphology = agent.morphology_space.sample()  # this is where you would insert your morph search algorithm
    agent.morph(morphology)
    observation, info = env.reset()

    for _ in range(500): # policy training
        action = agent.action_space.sample()
        observation, reward, terminated, info = env.step(action) # this is where you would insert your policy
        env.reset(terminated)
        env.render()
env.close()
```


## Customize your own world


Run a simple training process by the following command (set `headless=True` to disable the viewer):

`python train.py task=Dog morph=PSOSearch train=MorphTensor headless=False`

## Settings

We use Hydra to manage configurations
 * https://hydra.cc/docs/intro/

Following main options are currently supported:

```
task: Kangaroo | Raptor | Dog | 
terrain: Flat | Uphill | Downhill | 
train: SinglePPO | MorphTensor 
morph: Fix | PSOSearch
```

See more arguments in `inrl/cfg`.