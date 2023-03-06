

from morphgym.agents.base.agent import Agent
from morphgym.agents.base.xml_agent import XMLAgent


from morphgym.agents.loc2d.dog2dagent import Dog2DAgent
from morphgym.agents.loc2d.raptor2dagent import Raptor2DAgent
from morphgym.agents.loc2d.kangaroo2dagent import Kangaroo2DAgent


# from morphgym.agents.unimal.unimal_agent import UnimalAgent



agent_map = {
    'Agent': Agent,
    "XML": XMLAgent,
    "Dog2D": Dog2DAgent,
    "Raptor2D":Raptor2DAgent,
    "Kangaroo2D":Kangaroo2DAgent,
    # "Unimal":UnimalAgent,
}