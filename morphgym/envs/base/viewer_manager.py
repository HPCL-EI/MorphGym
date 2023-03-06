
import torch
from morphgym.utils import suppress_stdout_stderr

from typing import Callable

from isaacgym import gymapi
import sys
import torch



# create simplified key map
key_map = {}
for key,value in gymapi.KeyboardInput.__members__.items():
    key_map[key[4:]] = value


class ViewerManager(object):
    def __init__(self,gym,sim):
        self.gym , self.sim = gym,sim
        self.viewer_following = False
        self.enable_viewer_sync = True
        self.following_pos = None

        self.registed_event_dict = {}
        self.create_viewer()

    def create_viewer(self):
        # subscribe to keyboard shortcuts
        self._viewer = self.gym.create_viewer(
            self.sim, gymapi.CameraProperties())

        self.subscribe_keyboard_event("toggle_viewer_sync","V",self.toggle_viewer_sync)
        self.subscribe_keyboard_event("toggle_viewer_following","F",self.toggle_viewer_following)

        # set the camera position based on up axis
        self.cur_pos = gymapi.Vec3(0, -4.0, 1)
        self.cur_target = gymapi.Vec3(0, 0, 1)

        self.gym.viewer_camera_look_at(self._viewer, None, self.cur_pos, self.cur_target)


    def follow_pos(self,pos, pos_offset):
        """
        set a dynamic pos for the viewer to follow. Please make sure the pos is a View of your coordinate tensor.
        """
        self.viewer_following = True
        self.following_pos = pos
        self.following_pos_offset = pos_offset

    def toggle_viewer_sync(self):
        self.enable_viewer_sync = not self.enable_viewer_sync

    def toggle_viewer_following(self):
        if self.following_pos:
            self.viewer_following = not self.viewer_following

    def subscribe_keyboard_event(self, event_name:str,event_key:str,func:Callable) -> None:
        if event_key not in key_map:
            raise KeyError(f"Key '{event_key}' is unsupported! Please use one of the following keys: {list(key_map.keys())}")

        self.gym.subscribe_viewer_keyboard_event(self._viewer, key_map[event_key], event_name)
        self.registed_event_dict[event_name] = func

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        # if not self._viewer: return
        #
        # # update viewer pos
        # if self.viewer_following:
        #     self.cur_pos = gymapi.Vec3(self.following_pos[0] + self.following_pos_offset[0],
        #                                self.following_pos[1] + self.following_pos_offset[1],
        #                                self.following_pos[2] + self.following_pos_offset[2]),
        #     self.cur_target = gymapi.Vec3(self.following_pos[0]  + self.following_pos_offset[0],
        #                                   self.following_pos[1],
        #                                   self.following_pos[2])
        #
        #     self.gym.viewer_camera_look_at(
        #         self._viewer, None, self.cur_pos, self.cur_target)
        #
        # # check for window closed
        # if self.gym.query_viewer_has_closed(self._viewer):
        #     if 'close' in self.registed_event_dict:
        #         self.registed_event_dict['close']()
        #
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self._viewer):
            for event_name,func in self.registed_event_dict.items():
                if evt.action == event_name and evt.value > 0:
                    func()

        #fetch_results
        # self.gym.fetch_results(self.sim, True)

        if self._viewer:
            if self.gym.query_viewer_has_closed(self._viewer):
                sys.exit()

        # step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self._viewer, self.sim, True)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

        else:
            self.gym.poll_viewer_events(self._viewer)

