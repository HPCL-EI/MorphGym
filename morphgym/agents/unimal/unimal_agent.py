
# from .morph_unimal import BodyUnimal
from morphgym.agents.unimal.derl.envs.morphology import SymmetricUnimal
from morphgym.agents.base.agent1 import Agent

import gymnasium

from typing import (
    Any,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

T_cov = TypeVar("T_cov", covariant=True)


class UnimalMorphSpace(gymnasium.Space):
    """
    Unimal morph space.
    """
    def __init__(self,seed=0):
        super(UnimalMorphSpace, self).__init__(shape=(1,),seed=seed)

    def sample(self, mask: Optional[Any] = None) -> T_cov:
        self.unimal = SymmetricUnimal(
            id_=0,
            seed=0
        )
        self.unimal.mutate()
        path = self.unimal.save()
        return path

class UnimalAgent(Agent):
    """
    The Unimal space proposed in DERL.

    Since the space dimension is not the same, we need to provide the connection information.
    """
    def __init__(self,cfg):
        super().__init__(cfg)
        self.seed = cfg.get("seed",0)
        self.num_morph = cfg.get("num_morph",1)
        self.num_actor = cfg.get("num_actor",1)
        # self.morph_space = spaces.Box(-1,1,shape=(53,))
        # self.observation_space = spaces.Box(-np.inf,np.inf,shape=(85,))
        # self.action_space = spaces.Box(-1,1,shape=(20,))
        # self.morph_list = []

        self._morph_space = gymnasium.spaces.Tuple([UnimalMorphSpace() for _ in range(self.num_morph)], seed = 0)


    @property
    def observation_interface_names(self):
        return {
            "jacobian",
            "rigid_body_state"
        }


    @property
    def morph_space(self):
        return self._morph_space


    def get_morph_file(self,format='mjcf'):
        """
        get morphology file list for the
        """
        return self._morph_file

    def set_morph(self,morph):
        """
        Phase morph and get observation space, action space, and other information.
        """
        self.morph = morph
        self._morph_file = morph

