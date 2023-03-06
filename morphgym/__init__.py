import os

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

# block the output of isaac gym
with suppress_stdout_stderr():
    from isaacgym import gymapi



# envs
import morphgym.envs
from morphgym.envs.base.env import Env
from morphgym.envs.base.issac_gym_env import IssacGymEnv

# agents
import morphgym.agents
from morphgym.agents.base.agent import Agent

# morphology
from morphgym.morphology.morphology import Morphology



# make
from morphgym.envs import env_map
from morphgym.agents import agent_map

from morphgym.utils.input import load_cfg
import inspect
from morphgym.utils.Odict import ODict




def make(agent, env, config=None):
    """
    Entry to the MorphGym.

    param: agent str or class: str to use platform agent. class to use user.
    param: env str or class: str to use platform agent. class to use user.
    """

    # parse config
    if config is None:
        user_cfg = ODict({})
    elif isinstance(config, str):
        user_cfg = load_cfg(config)
    elif isinstance(config, dict):
        user_cfg = ODict(config)
    else:
        raise TypeError('config must be a path to the "config.yaml" file (string) or a dict (object).')

    cfg = ODict({
        'agent':{
            'single': {},
            'vector': {},
            'morphology': {
                'max_limbs': 12
            }
        },
        'env':{
            'single': {},
            'vector': {}
        }
    })
    cfg.update(user_cfg)

    # agent
    if isinstance(agent, str):
        agent_cls = agent_map[agent]
    elif inspect.isclass(agent) and Agent in agent.mro():
        agent_cls = agent
    else:
        raise TypeError('arg agent must be an agent name (string) file or an Agent (class).')
    agent = agent_cls(cfg.agent)

    # env
    if isinstance(env, str):
        env_cls = env_map[env]
    elif inspect.isclass(env) and Env in env.mro():
        env_cls = env
    else:
        raise TypeError('arg agent must be an env name (string) file or an Env (class).')
    env = env_cls(cfg.env, agent)

    return agent, env