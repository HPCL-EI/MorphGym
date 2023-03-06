

from gymnasium.spaces.sequence import Sequence
from gymnasium.spaces import Tuple,Dict,Box,Discrete
import numpy as np



class MorphologySpace(Tuple):
    def __init__(self, morphology_cfg=None):
        # if ()
        self.morphology_cfg = morphology_cfg
        self.max_limbs = morphology_cfg.max_limbs
        self.max_joints_per_limb = morphology_cfg.max_joints_per_limb
        raw_dict = {
            "parent_idx": Box(low=-1, high=np.inf, shape=(1,)),
            "idx": Box(low=-1, high=np.inf, shape=(1,)),
            "geom": Dict({
                "type": Box(low=0, high=1, shape=(1,)),
                "size": Box(low=-np.inf, high=np.inf, shape=(2,)),
                "mass": Box(low=0, high=np.inf, shape=(1,)),
            }),
            'joints': Tuple( Dict({
                            'axis': Box(low=-1, high=1, shape=(3,)),
                            'pos': Box(low=-np.inf, high=np.inf, shape=(3,)),
                            'range': Box(low=-np.inf, high=np.inf, shape=(2,)),
                            'gear': Box(low=0, high=np.inf, shape=(1,)),
                        }) for _ in range(self.max_joints_per_limb))
        }

        limb_dict = Dict(raw_dict)

        super(MorphologySpace, self).__init__([limb_dict for _ in range(self.max_limbs)])