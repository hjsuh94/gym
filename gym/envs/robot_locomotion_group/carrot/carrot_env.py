import numpy as np
import gym 
from gym import error, spaces, utils 
from gym.utils import seeding

from gym.envs.robot_locomotion_group.carrot.carrot_sim import CarrotSim
from gym.envs.robot_locomotion_group.carrot.carrot_rewards import lyapunov, image_transform

class CarrotEnv(gym.Env):
    metadata = {'render:modes': ['human']}

    def __init__(self):
        self.sim = CarrotSim()

        self.action_space = spaces.Box(
            low=-0.4,
            high=0.4, shape=(4,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0,
            high=1, shape=(32,32),
            dtype=np.float32
        )

    def step(self, action):
        current_image = self.sim.get_current_image()
        reward = lyapunov(image_transform(current_image))
        self.sim.update(action)
        next_image = self.sim.get_current_image()
        next_image = image_transform(next_image)

        print(reward)

        return next_image, reward, False, {}

    def reset(self):
        self.sim.refresh()
        current_image = self.sim.get_current_image()
        return image_transform(current_image)

    def render(self, mode='human'):
        self.sim.get_current_image()

    def close(self):
        pass
