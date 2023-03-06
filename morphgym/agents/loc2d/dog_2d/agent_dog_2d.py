
import numpy as np

import morphgym
from gymnasium import spaces


class AgentDog2D(morphgym.Agent):
    def __init__(self):
        super().__init__()
        self.morph_space = spaces.Box(-1,1,shape=(53,))
        self.observation_space = spaces.Box(-np.inf,np.inf,shape=(85,))
        self.action_space = spaces.Box(-1,1,shape=(20,))

    def set_morph(self,morph):
        pass
