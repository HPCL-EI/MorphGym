
import numpy as np

import morphgym

from gymnasium import spaces
from morphgym.agents.base.agent import Agent


class Dog2DAgent(Agent):
    """
    Dog 2D, only change the leg length and width.

    """
    def __init__(self):
        super().__init__()
        # self.morph_space = spaces.Box(-1,1,shape=(53,))
        # self.observation_space = spaces.Box(-np.inf,np.inf,shape=(85,))
        # self.action_space = spaces.Box(-1,1,shape=(20,))
        self.morph_list = []

    def set_morph(self,morph):

        # self.action_space = 1

        # from morphology get action and observation spaces.
        pass


    def sample_morph(self):
        unimal = SymmetricUnimal(0)
        unimal.mutate()
        return unimal.save()

    # self.morph_space = morphgym.space.UniqueSpace(sample_func = sample_unimal)
