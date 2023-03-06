
from morphgym.structure.data import data
from morphgym.morphology.data import MorphologyConfig



@data
class AgentConfig:
    num_morphologies: int = 1

    morphology = MorphologyConfig()



@data
class Space:
    action = None
    observation = None

@data
class AgentInfo:
    num_subenvs = None
    num_actors = None
    num_morphologies = None
    action_dim = None
    max_limbs = None
    max_joints_per_limb = None
    limb_observation_dim = 41
    xml_list: list = ()



