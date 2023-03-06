

from morphgym.structure.data import data




@data
class Space:
    action = None
    observation = None

    single_observation = None
    single_action = None

@data
class EnvInfo:
    num_subenvs = None
    device = 'cuda:0'



@data
class Buf:
    observation = None
    state = None
    joint_part_observation = None
    reward = None
    terminated = None
    action = None


@data
class TensorData:
    actor_root_state = None
    dof_state = None
    rigid_body_state = None
    dof_force = None


@data
class EnvConfig:
    num_actors: int = 1
    # observation_padding: bool = True



@data
class TensorView:
    actor_root_state = None
    dof_state = None
    rigid_body_state = None
    dof_force = None

    set_dof_state = None
    set_dof_pos = None
    set_dof_vel = None


@data
class TaskData:
    target = None
    potentials = None
    prev_potentials = None
    progress = None
    last_pos = None


@data
class ActorData:
    asset: list = ()
    subenv_ptr: list = ()
    handle: list = ()

    num_dofs = None
    num_actor_dofs = None

    num_actor_bodies = None

    gear = None
    dof_limits_lower = None
    dof_limits_upper = None

    initial_dof_pos = None
    initial_dof_vel = None
    initial_root_states = None

    env_dof_idx_start = None
    env_dof_idx_end = None

@data
class Mask:
    observation = None
    action = None
    dense_joint = None
    joint = None

    total_observation = None
    total_action = None
