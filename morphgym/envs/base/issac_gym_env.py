
import os
from gymnasium.spaces import Box

from morphgym import suppress_stdout_stderr
from torch import Tensor

with suppress_stdout_stderr():
    from isaacgym import gymapi
    from isaacgym import gymtorch

from morphgym.envs.base.env import Env
from morphgym.envs.base.viewer_manager import ViewerManager
from morphgym.envs.base.data import Buf, TensorData, TensorView, Mask,EnvInfo,Space,ActorData,TaskData

from morphgym.utils.torch_jit_utils import *
from morphgym.envs.base.data import EnvConfig
from morphgym.agents.base.agent import Agent

class IssacGymEnv(Env):
    gymapi = gymapi

    """
    This class is based on `Issac Gym <https://developer.nvidia.com/isaac-gym>`_ simulator.
    """
    def __init__(self, env_cfg: EnvConfig, agent: Agent):
        """
        new config:

        acquire_tensors (tuple):
            actor_root_state,
            dof_force,
            dof_state,
            force_sensor,
            jacobian,
            rigid_body_state,
        """
        super(IssacGymEnv, self).__init__(env_cfg, agent)
        self.gym = self._init_issacgym()
        self.sim = None

        self.dt = 1/60

        self.spacing = 5
        self.num_per_row = 16
        self.asset_path = None
        self.asset_file = None

        self.tensor = TensorData()
        self.tensor_view = TensorView()
        self.actor = ActorData()


        self.info.device = self.cfg.device
        
        self.acquire_tensors = [
            "actor_root_state",
            # "dof_force",
            "dof_state",
            # "force_sensor",
            # "jacobian",
            "rigid_body_state"
        ]

    # def _parse_issac_gym_cfg(self):
    #     self.info.device = self.cfg.get("device", "cuda:0")
    #     self.view = self.cfg.get("view", False)

    def _init_issacgym(self):
        # block the output of isaac gym
        with suppress_stdout_stderr():
            # optimization flags for pytorch JIT
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)

            gym = gymapi.acquire_gym()

            # basic sim params
            sim_params = gymapi.SimParams()
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.substeps = 2
            sim_params.gravity = gymapi.Vec3(*[0.0, 0.0, -9.81])
            sim_params.dt = 0.0166  # 1/60 s
            sim_params.use_gpu_pipeline = False if self.cfg.device == "cpu" else True

            # physx
            sim_params.physx.num_threads = 4  # Number of worker threads per scene used by PhysX - for CPU PhysX only.
            sim_params.physx.solver_type = 1  # 0: pgs, 1: tgs
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 0
            sim_params.physx.contact_offset = 0.01
            sim_params.physx.rest_offset = 0.0

            sim_params.physx.bounce_threshold_velocity = 0.2
            sim_params.physx.max_depenetration_velocity = 10.0
            sim_params.physx.default_buffer_size_multiplier = 5.0
            sim_params.physx.max_gpu_contact_pairs = 8388608  # 8*1024*1024
            sim_params.physx.num_subscenes = 4  # Splits the simulation into N physics scenes and runs each one in a separate thread

            sim_params.physx.use_gpu = False if self.cfg.device == "cpu" else True

            self.sim_params = sim_params

        print("lunched Issac Gym.")
        self.gym = gym
        return gym

    def create_sim(self):
        # create sim
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)

    def reset_agent(self):
        """
        put agents into the environment.

        :param :class:`morphgym.Agent` agent:
        """
        self.agent_info = self.agent.info

        self.info.num_subenvs = self.cfg.num_actors * self.agent.cfg.num_morphologies


    def create_subenvs(self):
        lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        asset_options.angular_damping = 1.0

        asset_list = []
        motor_efforts = []
        num_body_list = []
        num_dof_list = []
        # self.num_body_list = []
        # self.actuator_prop_list = []
        for asset_file  in self.agent.morphology.xml_list:
            path,name = os.path.split(asset_file)
            asset = self.gym.load_asset(self.sim, path, name, asset_options)
            asset_list.append(self.gym.load_asset(self.sim, path, name, asset_options))

            # get asset information
            actuator_props = self.gym.get_asset_actuator_properties(asset)
            motor_efforts.extend([prop.motor_effort for prop in actuator_props])

            num_body_list.append(self.gym.get_asset_rigid_body_count(asset))
            # actuator_prop_list.append(self.gym.get_asset_actuator_properties(asset))
            num_dof_list.append(self.gym.get_asset_dof_count(asset))

        self.actor.asset = asset_list


        start_pose = gymapi.Transform()
        # start_pose.p = gymapi.Vec3(*get_axis_params(0.44, self.up_axis_idx))
        start_pose.p = gymapi.Vec3(0,0,0.8)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
                                           device=self.info.device)

        self.torso_index = 0
        # self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        # body_names = [self.gym.get_asset_rigid_body_name(ant_asset, i) for i in range(self.num_bodies)]
        # extremity_names = [s for s in body_names if "foot" in s]
        # self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.info.device)

        # create force sensors attached to the "feet"
        # extremity_indices = [self.gym.find_asset_rigid_body_index(ant_asset, name) for name in extremity_names]
        # sensor_pose = gymapi.Transform()
        # for body_idx in extremity_indices:
        #     self.gym.create_asset_force_sensor(ant_asset, body_idx, sensor_pose)

        self.actor.subenv_ptr = []
        self.actor.handle = []
        for actor_idx in range(self.cfg.num_actors):
            for morphology_idx in range(self.agent.cfg.num_morphologies):
                subenv_ptr = self.gym.create_env(
                    self.sim, lower, upper, self.num_per_row
                )
                actor_handle = self.gym.create_actor(subenv_ptr, asset_list[morphology_idx], start_pose, "actor", actor_idx * self.agent.cfg.num_morphologies + morphology_idx, 0)

                for j in range(num_body_list[morphology_idx]): # set color
                    self.gym.set_rigid_body_color(
                        subenv_ptr, actor_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

                self.actor.subenv_ptr.append(subenv_ptr)
                self.actor.handle.append(actor_handle)

        # dof_prop = self.gym.get_actor_dof_properties(subenv_ptr, ant_handle)
        # for j in range(self.num_dof):
        #     if dof_prop['lower'][j] > dof_prop['upper'][j]:
        #         self.dof_limits_lower.append(dof_prop['upper'][j])
        #         self.dof_limits_upper.append(dof_prop['lower'][j])
        #     else:
        #         self.dof_limits_lower.append(dof_prop['lower'][j])
        #         self.dof_limits_upper.append(dof_prop['upper'][j])

        # self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.info.device)
        # self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.info.device)

        # for i in range(len(extremity_names)):
        #     self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles[0],
        #                                                                           extremity_names[i])



    def prepare_issac_gym(self):
        # acquire tensors
        self._refresh_tensor_func_pool = []
        for tensor_name in self.acquire_tensors:
            if tensor_name == "jacobian":
                _tensor = getattr(self.gym, f"acquire_{tensor_name}_tensor")(self.sim,"actor")
                self.tensor[tensor_name] = gymtorch.wrap_tensor(_tensor)
                self._refresh_tensor_func_pool.append(getattr(self.gym,f"refresh_{tensor_name}_tensors"))
            else:
                _tensor = getattr(self.gym, f"acquire_{tensor_name}_tensor")(self.sim)
                self.tensor[tensor_name] = gymtorch.wrap_tensor(_tensor)
                self._refresh_tensor_func_pool.append(getattr(self.gym,f"refresh_{tensor_name}_tensor"))

        self.gym.prepare_sim(self.sim)


    def prepare_data(self):
        # values
        max_limbs = self.agent.morphology.cfg.max_limbs
        max_joints_per_limb = self.agent.morphology.cfg.max_joints_per_limb

        geom_dim = self.agent.morphology.info.geom_dim
        joint_dim = self.agent.morphology.info.joint_dim

        morphology_dim = geom_dim + max_joints_per_limb * joint_dim
        state_dim = 17

        num_subenvs = self.info.num_subenvs
        num_actors = self.cfg.num_actors
        num_morphologies = self.agent.cfg.num_morphologies


        # set tensor view
        self.tensor.dof_force = torch.zeros((num_subenvs, max_limbs * max_joints_per_limb), device=self.info.device)

        self.tensor_view.rigid_body_state = self.tensor.rigid_body_state.view(num_actors,-1,13)

        self.tensor_view.dof_state = self.tensor.dof_state.view(num_actors,-1,2)

        self.tensor_view.set_dof_state = self.tensor.dof_state.clone()
        self.tensor_view.set_dof_pos = self.tensor_view.set_dof_state[..., 0]
        self.tensor_view.set_dof_vel = self.tensor_view.set_dof_state[..., 1]



        # actor
        motor_efforts = []
        num_body_list = []
        num_dof_list = []
        for asset in self.actor.asset: # get asset information
            actuator_props = self.gym.get_asset_actuator_properties(asset)
            motor_efforts.extend([prop.motor_effort for prop in actuator_props])
            num_body_list.append(self.gym.get_asset_rigid_body_count(asset))
            num_dof_list.append(self.gym.get_asset_dof_count(asset))

        self.actor.gear = torch.tensor([motor_efforts], dtype=torch.float, device=self.info.device).\
            repeat(self.cfg.num_actors,1)

        self.actor.num_actor_dofs = []
        self.actor.num_actor_bodies = []
        for _ in range(num_actors):
            self.actor.num_actor_dofs.extend(num_dof_list)
            self.actor.num_actor_bodies.extend(num_body_list)

        self.actor.num_dofs = self.cfg.num_actors * sum(num_dof_list)

        # self.actor.env_dof_idx_start = (torch.arange(self.cfg.num_actors,device=self.info.device) * num_actors ).view(-1,1).repeat(1,num_actor_dofs)
        # self.actor.env_dof_idx_end = self.actor.env_dof_idx_start.clone()
        #
        dof_limits_lower = []
        dof_limits_upper = []
        for i in range(len(self.actor.subenv_ptr)): # dof limit
            subenv_ptr = self.actor.subenv_ptr[i]
            actor_handle = self.actor.handle[i]
            dof_prop = self.gym.get_actor_dof_properties(subenv_ptr, actor_handle)

            # dof_prop["driveMode"].fill(gymapi.DOF_MODE_EFFORT)

            for j in range(self.actor.num_actor_dofs[i]):
                if dof_prop['lower'][j] > dof_prop['upper'][j]:
                    dof_limits_lower.append(dof_prop['upper'][j])
                    dof_limits_upper.append(dof_prop['lower'][j])
                else:
                    dof_limits_lower.append(dof_prop['lower'][j])
                    dof_limits_upper.append(dof_prop['upper'][j])

        self.actor.dof_limits_lower = to_torch(dof_limits_lower, device=self.info.device)
        self.actor.dof_limits_upper = to_torch(dof_limits_upper, device=self.info.device)

        # actor init
        self.actor.initial_root_states = self.tensor.actor_root_state[:self.info.num_subenvs,:].clone()
        self.actor.initial_root_states[:self.info.num_subenvs, 7:13] = 0  # set lin_vel and ang_vel to 0


        self.actor.initial_dof_pos = torch.zeros_like(self.tensor_view.set_dof_pos, device=self.info.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.info.device)
        self.actor.initial_dof_pos = torch.where(self.actor.dof_limits_lower > zero_tensor, self.actor.dof_limits_lower,
                                           torch.where(self.actor.dof_limits_upper < zero_tensor, self.actor.dof_limits_upper,
                                                       self.actor.initial_dof_pos))

        self.actor.initial_dof_vel = torch.zeros_like(self.tensor_view.set_dof_vel, device=self.info.device, dtype=torch.float)

        # space
        self.space.action = Box(low=-1, high=1, shape=(self.info.num_subenvs,
                                                       self.agent.info.action_dim))


        # set mask
        self.mask.observation = torch.tensor(self.agent.morphology.mask.limb,device=self.info.device)
        total_limbs = self.mask.observation.count_nonzero().item()
        self.mask.total_observation = self.mask.observation.repeat(num_actors,1)

        self.mask.joint = torch.tensor(self.agent.morphology.mask.joint,device=self.info.device)
        self.mask.dense_joint = torch.tensor(self.agent.morphology.mask.dense_joint,device=self.info.device)

        self.mask.action = self.agent.morphology.mask.joint
        self.mask.total_action = self.mask.joint.repeat(num_actors,1)


        # set buffer
        self.buf.observation = torch.zeros((num_actors,
                                            self.agent.cfg.num_morphologies,
                                            self.agent.morphology.cfg.max_limbs,
                                            self.agent.info.limb_observation_dim), device=self.info.device)
        self.buf.state = torch.zeros((num_actors, num_morphologies, max_limbs, 17), device=self.info.device)
        self.buf.joint_part_observation = torch.zeros((num_actors, total_limbs * max_joints_per_limb, 2,), device=self.info.device)
        self.buf.reward = torch.zeros((num_subenvs,1), device=self.info.device)
        self.buf.terminated = torch.zeros((num_subenvs,), device=self.info.device)
        self.buf.action = torch.zeros((self.info.num_subenvs, self.agent.info.action_dim),device=self.info.device)

        # pre-fill buffer
        self.buf.observation[:, :, :, state_dim:] = torch.tensor(np.array(self.agent.morphology.flat)).view(num_morphologies, max_limbs, morphology_dim).repeat(num_actors, 1, 1, 1)

        self.task.potentials =  torch.tensor([-1000. / self.dt], device=self.info.device).repeat(num_subenvs)
        self.task.prev_potentials = self.task.potentials.clone()
        self.task.progress = self.buf.terminated.clone()
        self.task.last_pos = self.buf.terminated.clone()  # reset no moving agents
        self.task.target =  torch.tensor([1000., 0, 0], device=self.info.device).repeat((num_subenvs, 1))



        self.space.single_observation = Box(low=-1, high=1,
                                            shape=(self.agent.morphology.cfg.max_limbs * self.agent.info.limb_observation_dim,))


    def step(self, action):
        # pre physics step
        self.buf.action[:] = to_torch(action,device=self.info.device)
        self.tensor.dof_force = self.buf.action.view(self.cfg.num_actors,
                                                     self.agent.cfg.num_morphologies,
                                                     self.agent.info.action_dim)[:,self.mask.joint] * self.actor.gear

        _force_tensor = gymtorch.unwrap_tensor(self.tensor.dof_force)
        self.gym.set_dof_actuation_force_tensor(self.sim, _force_tensor)

        # simulate
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        if self.cfg.view:
            self.viewer_manager.render()

        # post physics step
        self.task.progress += 1

        env_ids = self.buf.terminated.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        # self.reset_idx(torch.tensor([0,2,3],dtype=torch.long,device=self.info.device))

        self.refresh_tensors()

        # return status
        return self.observation(), self.reward(), self.terminated() ,self.get_info()

    def get_info(self):
        return self.info

    def refresh_tensors(self):
        for func in self._refresh_tensor_func_pool:
            func(self.sim)


    # viewer
    def create_viewer(self):
        """
        create and prepare the Viewer.
        """

        if not self.cfg.view:
            return

        viewer_manager = ViewerManager(self.gym, self.sim)
        viewer_manager.subscribe_keyboard_event("close", "Q", self.close)
        viewer_manager.subscribe_keyboard_event('reset', "R", self.reset)
        self.viewer_manager = viewer_manager
        return viewer_manager

    def reset(self,*,seed = None,options = None,**kwargs):
        if self.sim is None:
            # self.gym.destroy_sim(self.sim)
            # self.gym.destroy_viewer self.viewer_manager
            self.reset_agent()
            self.create_sim()
            self.create_subenvs()
            self.create_viewer()
            self.create_scenario()
            self.prepare_issac_gym()
            self.refresh_tensors()
            self.prepare_data()

        # self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        #
        # # asymmetric actor-critic
        # if self.num_states > 0:
        #     self.obs_dict["states"] = self.get_state()

        self.reset_idx(torch.arange(self.info.num_subenvs,device=self.info.device))
        return self.buf.observation


    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU

        positions = torch_rand_float(-0.2, 0.2, (self.actor.num_dofs,1), device=self.info.device)
        velocities = torch_rand_float(-0.1, 0.1, (self.actor.num_dofs,1), device=self.info.device)

        self.tensor_view.set_dof_pos = tensor_clamp(self.actor.initial_dof_pos + positions,
                                                    self.actor.dof_limits_lower,
                                                    self.actor.dof_limits_upper)
        self.tensor_view.set_dof_vel = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.actor.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.tensor_view.set_dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.task.target[env_ids] - self.actor.initial_root_states[env_ids, 0:3]
        to_target[:, 2] = 0.0
        self.task.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.task.potentials[env_ids] = self.task.prev_potentials[env_ids].clone()
        self.task.progress[env_ids] = 0
        self.task.last_pos[env_ids] = 0

        self.buf.terminated[env_ids] = 0
        self.buf.reward[env_ids] = 0


    def close(self):
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
        self.agent.close()



    def observation(self):
        raise NotImplementedError

    def reward(self):
        raise NotImplementedError

    def terminated(self):
        raise NotImplementedError

    def create_scenario(self):
        raise NotImplementedError


    @property
    def observation_space(self):
        return self.space.observation

    @property
    def action_space(self):
        return self.space.action