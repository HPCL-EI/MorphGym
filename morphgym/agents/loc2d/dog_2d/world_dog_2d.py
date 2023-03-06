
import morphgym
from .agent_dog_2d import AgentDog2D


class Dog2D(morphgym.World):
    def __init__(self):
        super().__init__(num_envs=1)
        self.agent = AgentDog2D()

    def reset(self):
        observation = None
        info = None

        return observation, info

    def reset_idx(self,terminated):
        return None

    def step(self,action):
        observation = None
        reward = None
        terminated = None
        info = None

        return observation, reward, terminated, info

    def render(self):
        pass

    def close(self):
        pass