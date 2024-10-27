import math

import numpy as np
from gymnasium import spaces, utils

from miniworld.entity import Box, MeshEnt
from miniworld.miniworld import MiniWorldEnv

from typing import Optional, Tuple
from gymnasium.core import ObsType


from miniworld.entity import Agent

from pyglet.window import key
import pyglet

import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

class WallGap(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Outside environment with two rooms connected by a gap in a wall. The
    goal is to go to a red box within as little steps as possible.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +(1 - 0.2 * (step_count / max_episode_steps)) when box reached

    ## Arguments

    ```python
    env = gym.make("MiniWorld-WallGap-v0")
    ```
    """

    def __init__(self, 
        time_id, 
        max_episode_time_step=250, 
        rendering=False, 
        wait_for_keypress=False, **kwargs):


        print(f"time_id: {time_id}")

        self.rendering = rendering
        
        # Select time
        if time_id == 0:
            self.loc_sky_color = [1.0, 0.75, 0.4]
            self.loc_light_amb = [1.0, 0.75, 0.4]
            self.loc_light_color = [1.0, 0.75, 0.4]
            self.wall_tex = "brick_wall"
            self.floor_tex = "concrete_tiles"
            self.road_tex = "asphalt"

        if time_id == 1:
            self.loc_sky_color = [0.4, 0.5, 1.0]
            self.loc_light_amb = [0.4, 0.5, 1.0]
            self.loc_light_color = [0.4, 0.5, 1.0]
            self.wall_tex = "concrete"
            self.floor_tex = "grass"
            self.road_tex = "asphalt"

        if time_id == 2:
#             self.loc_sky_color = [0.4, 0.75, 0.4]
#             self.loc_light_amb = [0.4, 0.75, 0.4]
#             self.loc_light_color = [0.4, 0.75, 0.4]

            self.loc_sky_color = [1.0, 0.5, 1.0]
            self.loc_light_amb = [1.0, 0.5, 1.0]
            self.loc_light_color = [1.0, 0.5, 1.0]
            self.wall_tex = "cinder_blocks"
            self.floor_tex = "slime"
            self.road_tex = "asphalt"

        if time_id == 3:
            self.loc_sky_color = [0.0, 0.2, 0.4]
            self.loc_light_amb = [0.0, 0.2, 0.4]
            self.loc_light_color = [0.0, 0.2, 0.4]
            self.wall_tex = "brick_wall"
            self.floor_tex = "concrete_tiles"
            self.road_tex = "asphalt"
        
        # print(f"time_id: {time_id}")
        # print(f"self.loc_sky_color: {self.loc_sky_color}")
        # print(f"self.loc_light_amb: {self.loc_light_amb}")
        # print(f"self.loc_light_color: {self.loc_light_color}")
        # print(f"self.wall_tex: {self.wall_tex}")
        # print(f"self.floor_tex: {self.floor_tex}")
        # print(f"self.road_tex: {self.road_tex}")
        # else:
        #     raise ValueError(f"Invalid time_id: {time_id}")

        MiniWorldEnv.__init__(self, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)
        
        # Create variables used in human interface
        self.stop_simulation = False  #  the stop_simulation flag will be set to True if user wants to interrupt the simulation
        self.key_pressed = {key.LEFT: False, key.RIGHT: False, key.UP: False}
        self.press_event = False
        self.wait_for_keypress = wait_for_keypress if self.rendering else False


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """
        super().reset(seed=seed)

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Agent()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []

        # Generate the world
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.np_random if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(
            rand, self, ["sky_color", "light_pos", "light_color", "light_ambient"]
        )
        
        # NOTE: added
        self.sky_color = self.loc_sky_color
        self.light_ambient = self.loc_light_amb
        self.light_color = self.loc_light_color
#         print(self.light_pos)

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max("forward_step")

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # print(f"self.wall_segs: {self.wall_segs}")

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = self.render_obs()

        # Return first observation
        return obs, {}
    


    def reinit(self, max_steps, wait_for_keypress):
        self.wait_for_keypress = wait_for_keypress if self.rendering else False
        if self.rendering:
            # Init
            self.render_mode = 'pyglet'
            super().render()

            # Decorate key interfaces
            self.on_key_press = self.unwrapped.window.event(self.on_key_press)
            self.on_key_release = self.unwrapped.window.event(self.on_key_release)
            
            self.stop_simulation = False
            
            self.press_event = False
            for key, value in self.key_pressed.items():
                self.key_pressed[key] = False

        self.max_episode_steps = max_steps
        
        return self.reset()


    def _gen_world(self):
        room0 = self.add_rect_room(
            min_x=-5, 
            max_x=5, 
            min_z=0.5, 
            max_z=6,
            wall_tex=self.wall_tex, 
            floor_tex=self.floor_tex, 
            no_ceiling=True
        )
        room1 = self.add_rect_room(
            min_x=-5, 
            max_x=5, 
            min_z=-6, 
            max_z=-0.5,
            wall_tex=self.wall_tex, 
            floor_tex=self.floor_tex, 
            no_ceiling=True
        )
        self.connect_rooms(room0, room1, min_x=-2.5, max_x=2.5)

        self.box = self.place_entity(Box(color="red"), room=room1)

        self.place_entity(
            MeshEnt(mesh_name="building", height=30),
            pos=np.array([30, 0, 30]),
            dir=-math.pi,
        )

        self.place_agent(room=room0)

    def _collision_with_wall(self, agent_pos):
        """
        Check if the agent collides with any of the wall segments.

        Parameters:
        agent_pos (tuple): The position of the agent as (x, z).

        Returns:
        bool: True if the agent collides with any wall, False otherwise.
        """
        wall_segments = [ # 12 points
            [[5.0, 0.0, 0.5], [5.0, 0.0, 6.0]],
            [[2.5, 0.0, 0.5], [5.0, 0.0, 0.5]],
            [[-5.0, 0.0, 0.5], [-2.5, 0.0, 0.5]],
            [[-5.0, 0.0, 6.0], [-5.0, 0.0, 0.5]],
            [[5.0, 0.0, 6.0], [-5.0, 0.0, 6.0]],
            [[5.0, 0.0, -6.0], [5.0, 0.0, -0.5]],
            [[-5.0, 0.0, -6.0], [5.0, 0.0, -6.0]],
            [[-5.0, 0.0, -0.5], [-5.0, 0.0, -6.0]],
            [[-2.5, 0.0, -0.5], [-5.0, 0.0, -0.5]],
            [[5.0, 0.0, -0.5], [2.5, 0.0, -0.5]],
            [[-2.5, 0.0, 0.5], [-2.5, 0.0, -0.5]],
            [[2.5, 0.0, -0.5], [2.5, 0.0, 0.5]]
        ]

        agent_x, _, agent_z = agent_pos

        agent_radius = 0.5  # Assuming a radius for the agent, slightly increased for better collision handling

        # Iterate through each wall segment
        for wall in wall_segments:
            start, end = np.array(wall[0]), np.array(wall[1])
            p0 = np.array([start[0], start[2]])
            p1 = np.array([end[0], end[2]])
            agent_point = np.array([agent_x, agent_z])

            wall_vector = p1 - p0
            agent_vector = agent_point - p0
            wall_length_squared = np.dot(wall_vector, wall_vector)

            if wall_length_squared == 0:
                # The wall is a point rather than a line segment
                distance = np.linalg.norm(agent_vector)
            else:
                # Project the agent vector onto the wall vector
                t = np.clip(np.dot(agent_vector, wall_vector) / wall_length_squared, 0, 1)
                projection = p0 + t * wall_vector
                distance = np.linalg.norm(agent_point - projection)

            # Check if the agent is within a collision distance
            if distance <= agent_radius:
                # print(f"collision with wall: {wall}")
                return True  # Collision detected

        return False  # No collision detected


    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        #  collision with wall
        if self._collision_with_wall(self.agent.pos):
            reward -= 2/self.max_episode_steps
            truncation = True

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info

    # def step(self, action):
    #     obs, reward, termination, truncation, info = super().step(action)
        
    #     agent_pos = np.array([self.agent.pos[0], self.agent.pos[2]])
    #     box_pos = np.array([self.box.pos[0], self.box.pos[2]])
        
    #     # Check if there's a wall between agent and goal
    #     wall_blocks_path = self._wall_blocks_path(agent_pos, box_pos)
        
    #     if wall_blocks_path:
    #         # If wall blocks the path, reward moving towards the gap
    #         gap_center = np.array([0, 0])
    #         distance_to_gap = np.linalg.norm(agent_pos - gap_center)
    #         reward = -0.01 * distance_to_gap
    #     else:
    #         # If direct path is available, reward moving towards goal
    #         distance_to_goal = np.linalg.norm(agent_pos - box_pos)
    #         reward = -0.01 * distance_to_goal
        
    #     if self._collision_with_wall(self.agent.pos):
    #         reward = -100.0
    #         truncation = True

    #     if self.near(self.box):
    #         reward += 1.0
    #         termination = True

    #     return obs, reward, termination, truncation, info
    
    def _wall_blocks_path(self, start_pos, end_pos):
        """Check if the wall blocks the direct path between start and end positions"""
        # If start and end are in different chambers (one z>0 and other z<0),
        # and the line between them doesn't pass through the gap
        if (start_pos[1] * end_pos[1] < 0):  # Points are on opposite sides of wall
            intersection_x = start_pos[0] + (start_pos[1] * (end_pos[0] - start_pos[0])) / (start_pos[1] - end_pos[1])
            return abs(intersection_x) > 2.5  # True if intersection point is outside gap
        
        return False  # No wall blocking if points are in same chamber or path goes through gap
    
    def render(self, mode='rgb_array'):
        if self.rendering:
            self.render_mode = 'pyglet'
            out = super().render()
#             out = super().render()#mode='pyglet')
            self.update_pyglet() # update pyglet info and capture keys
        else:
            self.render_mode = mode
            out = super().render()
#             out = super().render()#mode=mode)
#         print(out)
        return out

    def on_key_press(self, symbol, modifier):
        if symbol == key.ESCAPE: 
            self.stop_simulation = True  # set 'quit' flag if ESCAPE key is pressed
        self.key_pressed[symbol] = True
        self.press_event = True
    
    def on_key_release(self, symbol, modifier):
        self.key_pressed[symbol] = False
        self.press_event = False

    def update_pyglet(self):
        self.window.flip()
        while True:
            pyglet.clock.tick()
            self.window.dispatch_events()
            if (not self.wait_for_keypress) or self.press_event:
                break

    def close(self):
        super().close()
