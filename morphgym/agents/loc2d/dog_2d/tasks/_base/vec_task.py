# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the element of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import abc
import operator
import random
import sys
from abc import ABC
from copy import deepcopy
from typing import Dict, Any, Tuple

import gym
import numpy as np
import torch
from gym import spaces
from isaacgym import gymapi
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, \
    apply_random_samples, check_buckets, generate_random_samples


class Env(ABC):
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int, headless: bool):
        """Initialise the base.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = config.get("rl_device", "cuda:0")

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config.video.enable
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_agents = config["base"].get("numAgents", 1)  # used for multi-base environments
        self.num_observations = config["base"]["numObservations"]
        self.num_states = config["base"].get("numStates", 0)
        self.num_actions = config["base"]["numActions"]

        self.control_freq_inv = config["base"].get("controlFrequencyInv", 1)

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = config["base"].get("clipObservations", np.Inf)
        self.clip_actions = config["base"].get("clipActions", np.Inf)

        self.max_frames = config["video"].get("max_frames", None)
        if self.max_frames is None:
            self.max_frames = 99999999999

        self.cur_frames = 0

    @abc.abstractmethod
    def _allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_observations


class VecTask(Env):

    def __init__(self, config, sim_device, graphics_device_id, headless):
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """
        super().__init__(config, sim_device, graphics_device_id, headless)

        self.sim_params = self.__parse_sim_params(self.cfg["sim"]["physics_engine"], self.cfg["sim"])
        if self.cfg["sim"]["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["sim"]["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        self.video_pos = self.cfg.video.pos

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()

        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False
        self._create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.viewer_inited = False
        self._set_viewer()

        # video
        self.camera_inited = False
        self._set_camera()

        self._allocate_buffers()

        self.obs_dict = {}

    def _set_viewer(self):
        """Create the viewer."""

        self.viewer = None

        # if self.config.make_video:
        #     self.num_pics = 0
        #     if os.path.exists('./visual/video/pictures'):
        #         shutil.rmtree('./visual/video/pictures')
        #     os.mkdir('./visual/video/pictures')

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_C, "toggle_viewer_following")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_F, "toggle_fsm_enable")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "reset")

            sim_params = self.gym.get_sim_params(self.sim)
            # set the camera position based on up axis

            if self.viewer != None:
                if self.viewer_following:
                    cam_pos = gymapi.Vec3(0, -4.0, 1)
                    cam_target = gymapi.Vec3(0, 0, 1)
                else:
                    cam_pos = gymapi.Vec3(*self.video_pos)
                    cam_target = gymapi.Vec3(self.video_pos[0], self.video_pos[1] - 1, self.video_pos[2])

                    # cam_pos = gymapi.Vec3(0, 45875.2, 1)
                    # cam_target = gymapi.Vec3(0, 0, 1)

                    # cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
                    # cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
                self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # if sim_params.up_axis == gymapi.UP_AXIS_Z:
            #     cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            #     cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            # else:
            #     cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
            #     cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def _set_camera(self):

        """Create cameras to monitor an base"""
        if self.cfg.video.enable:
            import cv2
            self.cv2 = cv2
            # self.camera_pos = [1., -4, 1]

            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 75.0
            camera_props.width = self.cfg.video.camera_size[0]
            camera_props.height = self.cfg.video.camera_size[1]
            camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props)
            self.camera = camera_handle

            # attach to the base
            # local_transform = gymapi.Transform()
            # local_transform.p = gymapi.Vec3(*self.config.video.pos)
            # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(*self.config.video.rotate_axis), np.radians(self.config.video.rotate_angle))
            # root_rigid = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], 'root')
            # self.gym.attach_camera_to_body(camera_handle, self.envs[0], root_rigid, local_transform, gymapi.FOLLOW_POSITION)

            # init the video writer
            output_path = f'{self.root_path}/{self.cfg.video.output_path}'
            output_file = f'{output_path}/{self.cfg.video.file_name}'
            print(f'The camera video will be create in: {output_file}')
            self.camera_video_writer = self.cv2.VideoWriter(output_file, self.cv2.VideoWriter_fourcc(*'mp4v'),
                                                            self.cfg.video.fps,
                                                            (self.cfg.video.camera_size[0],
                                                             self.cfg.video.camera_size[1]), True)

    def step_graphics(self):
        # fetch results
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        # step graphics
        if self.enable_viewer_sync or self.cfg.video.enable:
            self.gym.step_graphics(self.sim)

            if not self.headless:
                self.gym.draw_viewer(self.viewer, self.sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)
        else:
            self.gym.poll_viewer_events(self.viewer)

    def render_viewer(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            if self.viewer_following or not self.viewer_inited:
                if self.viewer_following:
                    self.gym.viewer_camera_look_at(
                        self.viewer, None, gymapi.Vec3(self.root_states[0, 0] + self.cfg.video.pos[0],
                                                       self.root_states[0, 1] + self.cfg.video.pos[1],
                                                       self.root_states[0, 2] + self.cfg.video.pos[2]),
                        gymapi.Vec3(self.root_states[0, 0] + self.cfg.video.pos[0], self.root_states[0, 1],
                                    self.root_states[0, 2]))
                else:
                    self.gym.viewer_camera_look_at(
                        self.viewer, None,
                        gymapi.Vec3(*self.video_pos),
                        gymapi.Vec3(self.video_pos[0], self.video_pos[1] + 1,
                                    self.video_pos[2]))

                if not self.viewer_inited:  self.viewer_inited = True
            # self.gym.set_camera_location(self.viewer_handle,self.envs[0],
            #                              gymapi.Vec3(self.root_states[0,0],self.root_states[0,1] + self.config.video.pos[1],self.config.video.pos[2]),
            #                              gymapi.Vec3(self.root_states[0,0],0,self.config.video.pos[2]))

            v_size = self.gym.get_viewer_size(self.viewer)

            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                self.close()
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "toggle_viewer_following" and evt.value > 0:
                    self.viewer_following = not self.viewer_following
                elif evt.action == "toggle_fsm_enable" and evt.value > 0:
                    self.fsm_enable = not self.fsm_enable
                elif evt.action == "reset" and evt.value > 0:
                    self.reset_idx(torch.arange(self.num_envs))

    def render_camera(self):
        if self.cfg.video.enable:
            if self.viewer_following or not self.camera_inited:
                if self.viewer_following:
                    self.gym.set_camera_location(self.camera, self.envs[0],
                                                 gymapi.Vec3(self.root_states[0, 0] + self.cfg.video.pos[0],
                                                             self.root_states[0, 1] + self.cfg.video.pos[1],
                                                             self.root_states[0, 2] + self.cfg.video.pos[2]),
                                                 gymapi.Vec3(self.root_states[0, 0] + self.cfg.video.pos[0],
                                                             self.root_states[0, 1], self.root_states[0, 2]))
                else:
                    self.gym.set_camera_location(self.camera, self.envs[0],
                                                 gymapi.Vec3(*self.video_pos),
                                                 gymapi.Vec3(self.video_pos[0], self.video_pos[1] + 1,
                                                             self.video_pos[2]))

                if not self.camera_inited: self.camera_inited = True

            self.gym.render_all_camera_sensors(self.sim)
            imorphgym = self.gym.get_camera_image(self.sim, self.envs[0], 0, gymapi.IMAGE_COLOR)
            imorphgym = imorphgym.reshape(self.cfg.video.camera_size[0], self.cfg.video.camera_size[1], 4)
            imorphgym = self.cv2.cvtColor(imorphgym, self.cv2.COLOR_RGBA2BGR)
            self.camera_video_writer.write(imorphgym)
            self.cur_frames += 1
            if self.cur_frames >= self.max_frames:
                print(f'Max frames!')
                self.compute_reward = self.finish_rewards

    def finish_rewards(self):
        self.reset_buf[:] = 1

    def _allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def _create_gym_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = self.gym.init_issacgym(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def get_state(self):
        """Returns the state buffer of the environment (the priviledged observations for asymmetric training)."""
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.gym.simulate(self.sim)
            self.step_graphics()
            self.render_viewer()
            self.render_camera()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # fill time out buffer
        self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1,
                                       torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.cur_obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.cur_obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        # print(gymapi.Quat(*self.root_states[0,3:7]).to_euler_zyx())
        # print(self.obs_buf[:, 84])
        #
        # torch.set_printoptions(precision=4, sci_mode=False,linewidth=3000)
        # print(self.contact_tensor[0,:])

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def close(self):
        if self.cfg.video.enable:
            self.camera_video_writer.release()

        self.gym.destroy_sim(self.sim)

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)

        return actions

    def reset_idx(self, env_idx):
        """Reset environment with indces in env_idx. 
        Should be implemented in an environment code inherited from VecTask.
        """
        pass

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        self.obs_dict["obs"] = torch.clamp(self.cur_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def reset_done(self):
        """Reset the environment.
        Returns:
            Observation dictionary, indices of environments being reset
        """
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)

        self.obs_dict["obs"] = torch.clamp(self.cur_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params

    """
    Domain Randomization methods
    """

    def get_actor_params_info(self, dr_params: Dict[str, Any], env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """

        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name + '_' + str(prop_idx) + '_' + attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0 * float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name + '_' + str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    def _apply_randomizations(self, dr_params):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """

        # If we don't have a randomization frequency, randomize every step
        if not self.randomize: return

        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf),
                                    torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[
                    nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[
                    nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                                    min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                             (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                                  (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr,
                                                                 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr,
                                                                 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        for actor, actor_properties in dr_params["actor_params"].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1),
                                                                      random.uniform(0, 1)))
                        continue
                    if prop_name == 'scale':
                        setup_only = prop_attrs.get('setup_only', False)
                        if (setup_only and not self.sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(attr_randomization_params, 1,
                                                             self.last_step, None)
                            og_scale = 1
                            if attr_randomization_params['operation'] == 'scaling':
                                new_scale = og_scale * sample
                            elif attr_randomization_params['operation'] == 'additive':
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get('setup_only', False)
                                if (setup_only and not self.sim_initialized) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                            extern_sample, extern_offsets[env_id], p, attr)
                                    apply_random_samples(
                                        p, og_p, attr, attr_randomization_params,
                                        self.last_step, smpl)
                                else:
                                    set_random_properties = False
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            setup_only = attr_randomization_params.get('setup_only', False)
                            if (setup_only and not self.sim_initialized) or not setup_only:
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], prop, attr)
                                apply_random_samples(
                                    prop, self.original_props[prop_name], attr,
                                    attr_randomization_params, self.last_step, smpl)
                            else:
                                set_random_properties = False

                    if set_random_properties:
                        setter = param_setters_map[prop_name]
                        default_args = param_setter_defaults_map[prop_name]
                        setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False

    @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""
