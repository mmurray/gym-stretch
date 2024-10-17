# gym-stretch

A gym environment for Hello Robot Stretch


## Installation

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n stretch python=3.10 && conda activate stretch
```

Install gym-stretch:
```bash
pip install gym-stretch
```


## Quickstart

```python
# example.py
import imageio
import gymnasium as gym
import numpy as np
import gym_stretch

env = gym.make("gym_stretch/StretchLiftBlock-v0")
observation, info = env.reset()
frames = []

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)
```
