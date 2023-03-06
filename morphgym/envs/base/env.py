"""
Agent class controls all actors in the simulator.
"""
# from omegaconf import DictConfig


import gymnasium
from morphgym.agents.base.agent import Agent
from morphgym.envs.base.data import EnvConfig
from morphgym.envs.base.data import Buf, TensorData, TensorView, Mask,EnvInfo,Space,ActorData,TaskData

class Env(gymnasium.Env):
    """
    :super: `gymnasium.Env <https://gymnasium.farama.org/api/env/#gymnasium.Env>`_.

    We don't use  because we want to set different observation and action space for each agent.

    The Env in MorphGym has some features:

    * It naturally supports parallel simulation of multiple embodied agent.
    Unlike `gymnasium.vector.VectorEnv <https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv>`_.,
    it supports agent with different morphologies.
    For single morphology simulation, just pass a single agent using :py:meth:`set_agent()`.

    Environment dynamics (Simulation): terrains.
    Task performance: rewards.
    Simulator corresponding output: viewer, camera, images, information.
    """

    def __init__(self, env_cfg: EnvConfig, agent: Agent):
        self.cfg: EnvConfig = env_cfg
        self.agent: Agent = agent
        self.info = EnvInfo()
        self.buf = Buf()
        self.mask = Mask()
        self.space = Space()
        self.task = TaskData()


    def reset_agent(self):
        """
        reset agent with current morphologies.
        """
        pass

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        self.reset_agent()


        return None,None


    def close(self):
        self.agent.close()

    # @property
    # def info(self):
    #     return self.vec_env_cfg