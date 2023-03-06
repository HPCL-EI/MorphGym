


from morphgym.utils.torch_jit_utils import *

from morphgym.envs.base.issac_gym_env import IssacGymEnv
from morphgym.envs.base.data import EnvConfig
from morphgym.agents.base.agent import Agent


class Locomotion(IssacGymEnv):
    def __init__(self, env_cfg:EnvConfig, agent:Agent):
        super(Locomotion,self).__init__(env_cfg, agent)


    def observation(self):
        # joint
        self.buf.joint_part_observation[:, self.mask.dense_joint, :] = self.tensor_view.dof_state

        self.buf.state[:, self.mask.observation, :] = \
            torch.cat((self.tensor_view.rigid_body_state,
                       self.buf.joint_part_observation.view(self.cfg.num_actors, -1,
                                                            2 * self.agent.morphology.cfg.max_joints_per_limb)), dim=2)

        self.buf.observation[:, :, :, :17] = self.buf.state
        # body + joint
        return self.buf.observation

    def reward(self):
        self.buf.reward[:,:], self.task.potentials[:], self.task.prev_potentials[:] = compute_reward(
            self.tensor.actor_root_state[:,:3],
            self.task.target,
            self.task.potentials,
            self.task.prev_potentials,
            self.buf.action,
        )
        return self.buf.reward

    def terminated(self):
        self.buf.terminated[:], self.task.last_pos[:] = compute_terminated(
            self.tensor.actor_root_state[:,0],
            self.task.last_pos,
            self.buf.terminated,
            self.task.progress
        )

        return self.buf.terminated



@torch.jit.script
def compute_observations(root_states, targets, potentials,
                         inv_start_rot, dof_pos, dof_vel,
                         dof_limits_lower, dof_limits_upper, dof_vel_scale,
                         sensor_force_torques, actions, dt, contact_force_scale,
                         basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    rot = torch.stack((roll, pitch, yaw), dim=1)
    rot[rot > 3.1415926] -= 6.2831852

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs, num_dofs, 24, num_dofs(8)
    obs = torch.cat((torso_position[:, up_axis_idx].view(-1, 1), vel_loc, angvel_loc,
                     rot, angle_to_target.unsqueeze(-1),
                     up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled,
                     dof_vel * dof_vel_scale, sensor_force_torques.view(-1, 12) * contact_force_scale,
                     actions), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec



@torch.jit.script
def compute_reward(
        root_state,
        targets,
        potentials,
        prev_potentials,
        dof_force_tensor,
):
    # type: (Tensor, Tensor,Tensor, Tensor, Tensor) ->  Tuple[ Tensor,Tensor,Tensor]

    to_target = targets - root_state[:,0:3]
    to_target[:, 2] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) * 60

    # energy penalty for movement
    # electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 13 + ACT_DIM:13 + ACT_DIM * 2]), dim=-1)
    electricity_cost = torch.sum(torch.abs(dof_force_tensor), dim=1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward + alive_reward  \
                   - 0.0015 * electricity_cost

    return total_reward.unsqueeze(dim=1), potentials, prev_potentials_new



@torch.jit.script
def compute_terminated(
        cur_pos,
        last_pos,
        reset_buf,
        progress_buf
):
    # type: (Tensor, Tensor, Tensor,Tensor) -> Tuple[ Tensor,Tensor]

    # reset agents
    reset = torch.where(progress_buf >= 1023, torch.ones_like(reset_buf), reset_buf)
    # --  not moving forward
    mask = (progress_buf % 200) == 199
    ref_pos = torch.where(mask, last_pos, -torch.ones_like(last_pos))
    reset = torch.where((cur_pos - ref_pos) < 0.2, torch.ones_like(reset_buf), reset)

    new_last_pos_buff = torch.where(mask, cur_pos, last_pos)

    return reset, new_last_pos_buff
