"""
Agent class controls all actors in the simulator.
"""
# from omegaconf import DictConfig
from gymnasium.spaces import Box


import numpy as np
from morphgym.utils import ODict

from morphgym.agents.base.data import AgentConfig

from morphgym.agents.base import data


from collections import Iterable
from morphgym.structure.space.morphology_space import MorphologySpace
from lxml import etree
from gymnasium.spaces.utils import OrderedDict,flatten
from gymnasium.spaces import Tuple
from morphgym.utils import print_dict
import math

from morphgym.utils import ODict

from morphgym.morphology.morphology import Morphology


class Agent(object):
    """
    Superclass for Vector Agent, defining the morphology space, observation space and action space.

    Agent body morphology.
    Agent observation space, action space.
    """

    def __init__(self, agent_cfg: AgentConfig):
        # data
        self.cfg = agent_cfg
        self.info = data.AgentInfo()

        # self.num_morphologies = self.config.get("num_morphologies", 0)
        # for i in range(self.cfg.num_morphologies):
        #     self.append(agent_Class(agent_cfg.single))

        # object
        self.morphology: Morphology

    def morph(self, morphology: Morphology):
        """
        set (vector) agent's morphologies
        """
        morphology_cfg = morphology.cfg
        self.morphology: Morphology = morphology


        self.morphology_space = Tuple([MorphologySpace(morphology_cfg=morphology_cfg)
                                        for _ in range(self.cfg.num_morphologies)])
        # VectorMorphologySpace(
        #     num_morphologies=self.cfg.num_morphologies,
        #     morphology_cfg=morphology.cfg
        # )


        max_limbs = morphology_cfg.max_limbs
        max_joints_per_limb = morphology_cfg.max_joints_per_limb


        self.info.num_morphologies = self.cfg.num_morphologies
        # self.info.num_actors = self.cfg.num_actors
        # self.info.num_subenvs = self.cfg.num_morphologies * self.cfg.num_actors
        self.info.action_dim = max_limbs * max_joints_per_limb




    @property
    def observation_interface_names(self):
        return {}


    def close(self):
        pass