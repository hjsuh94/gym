import numpy as np
from gym import error, spaces, utils 
from gym.utils import seeding

from carrot_sim import CarrotSim
from carrot_reawrds import lyapunov

class CarrotEnv(gym.Env):
    metadata = {'render:modes': ['human']}

    def __init__(self):
        self.sim = CarrotSim()

        self.action_space = spaces.Box(
            low=-0.5,
            high=0.5, shape=(4,1),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255, shape=(500,500,3),
            dtype=np.uint8
        )

    def step(self, action):
        current_image = self.sim.get_current_image()
        reward = lyapunov(current_image)
        self.sim.update(action)
        next_image = self.sim.get_current_image()

        return next_image, reward, False, {}

    def reset(self):
        self.sim.reset()
        current_image = self.sim.get_current_image()
        return current_image

    def render(self, mode='human'):
        self.sim.get_current_image()

    def close(self):
        pass
