from morphgym.envs.locomotion.locomotion import Locomotion
from morphgym.envs.base.data import EnvConfig
from morphgym.agents.base.agent import Agent
from isaacgym import gymapi

class Plane(Locomotion):
    def __init__(self, env_cfg:EnvConfig, agent:Agent):
        super(Plane, self).__init__(env_cfg, agent)

    def create_scenario(self):
        self.create_ground()

    def create_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)
        print('created ground.')
