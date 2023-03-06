

from morphgym.agents.base.agent import Agent
from morphgym.agents.base.xml_agent import XMLAgent


from morphgym.agents.loc2d.dog2d import Dog2D
# from morphgym.agents.loc2d.raptor2dagent import Raptor2DAgent
# from morphgym.agents.loc2d.kangaroo2dagent import Kangaroo2DAgent


from morphgym.agents.unimal.unimal_agent import Unimal



agent_map = {
    'Agent': Agent,
    "XML": XMLAgent,
    "Dog2D": Dog2D,
    # "Raptor2D":Raptor2DAgent,
    # "Kangaroo2D":Kangaroo2DAgent,
    # "Unimal":UnimalAgent,
}