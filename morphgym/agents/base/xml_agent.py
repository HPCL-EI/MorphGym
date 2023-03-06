"""
XML Agent class controls all actors in the simulator.
"""

from collections import Iterable
from morphgym.agents.base.agent import Agent
from morphgym.structure.space.morphology_space import MorphologySpace
from lxml import etree
from gymnasium.spaces.utils import OrderedDict,flatten
from gymnasium.spaces import Tuple
from morphgym.utils import print_dict
import math


class XMLAgent(Agent):
    """
    The XMLAgent class, defining the body morphology, observation space and action space.

    Agent body morphology.
    Agent observation space, action space.
    """
    def __init__(self, agent_cfg):
        super(XMLAgent, self).__init__(agent_cfg)
        # tensors


