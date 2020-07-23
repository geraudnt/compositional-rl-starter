import gym
from gym import spaces
import numpy as np

class FixEnv(gym.core.Wrapper):
    """
    Wrapper to,
    - Terminate environment after reaching goal states. 
    - Make goal rewards binary.
    - Narrow down goal space by modifying goal states to show only the agent and the object infront of it.
    """

    def __init__(self, env):
        super().__init__(env)
        self.rmin = 0
        self.reward = self.rmin
        self.done = False
        self.agent_view_size = env.unwrapped.agent_view_size
    
    def remake(self):
        return self.env

    def reset(self):
        self.env.close()
        super().__init__(self.remake())
        obs = self.env.reset()
        self.reward = self.rmin           
        self.done = False
        return obs

    def step(self, action):  
        self.env.max_steps = float("inf")
        obs, reward, done, info = self.env.step(action)
        
        if self.done or done:
            ### Narrow down goal space
            self.env.agent_view_size = 3
            image_ = self.env.gen_obs()['image']
            image_[[0,2],:,:] = 0
            image_[:,0,:] = 0
            self.env.agent_view_size = self.agent_view_size

            image = np.zeros(shape=obs['image'].shape, dtype=obs['image'].dtype)
            ox = (image.shape[0]-image_.shape[0])//2
            oy = image.shape[1]-image_.shape[1]
            image[ox:ox+image_.shape[0],oy:oy+image_.shape[1],:] = image_
            obs['image'] = image

            ### Make goal rewards binary then terminate.
            if self.done:
                reward = self.reward
                self.reward = self.rmin
                done = self.done                
                self.done = False
            else:
                self.reward = reward
                reward = self.rmin
                self.done = done
                done = False
                                
        return obs, reward, done, {}

class FixEnvAndRGBImg(gym.core.Wrapper):
    """
    Wrapper to,
    - Terminate environment after reaching goal states. 
    - Make goal rewards binary.
    - Narrow down goal space by modifying goal states to show only the agent and the object infront of it.

    Warning: 
    - Do not use in conjunction with the FixEnv or RGBImg wrappers.
    - tile_size should be a multiple of 3.
    """

    def __init__(self, env, resize=False, obs_type='partial',tile_size=9):
        super().__init__(env)
        if resize:
            assert tile_size % 3 == 0
        
        self.done = False
        self.resize = resize
        self.obs_type = obs_type
        self.tile_size = tile_size
        self.agent_view_size = env.unwrapped.agent_view_size
        
        if self.obs_type == 'full':
            shape = (self.env.width, self.env.height, 3)          
        elif self.obs_type == 'partial':
            shape = env.observation_space.spaces['image'].shape
        else:
            assert False, "Unknown observation type. Please use obs_type='partial' or obs_type='full'"

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(shape[0] * tile_size, shape[1] * tile_size, 3),
            dtype='uint8'
        )
    
    def reset(self):
        obs = self.env.reset()
        obs['image'] = self.get_rgb_img(obs, self.obs_type)
        return obs

    def get_rgb_img(self, obs, obs_type, done=False):
        tile_size = self.tile_size
        if done:
            if self.resize:
                tile_size = self.agent_view_size*(self.tile_size//self.env.agent_view_size)
            else:
                s1 = (self.agent_view_size,self.agent_view_size,3)
                s2 = obs['image'].shape
                ox = (s1[0]-s2[0])//2
                oy = s1[1]-s2[1]
                image = np.zeros(shape=s1, dtype=obs['image'].dtype)
                image[ox:ox+s2[0],oy:oy+s2[1],:] = obs['image']
                obs['image'] = image
                
        if obs_type == 'full':
            rgb_img = self.env.render(
                mode='rgb_array',
                highlight=False,
                tile_size=tile_size
            )     
        else:
            rgb_img = self.env.get_obs_render(
                obs['image'],
                tile_size=tile_size
            )
        
        return rgb_img
        
    def step(self, action):  
        self.env.max_steps = float("inf")
        obs, reward, done, info = self.env.step(action)
        
        if self.done or done:
            ### Narrow down goal space
            self.env.agent_view_size = 3
            obs = self.env.gen_obs()
            obs['image'][[0,2],:,:] = 0
            obs['image'][:,0,:] = 0
            self.env.agent_view_size = self.agent_view_size
            rgb_img = self.get_rgb_img(obs, 'partial', done=done)
            self.env.agent_view_size = 3

            ### Make goal rewards binary then terminate.
            if self.done:
                reward = 1
                done = True
                self.done = False
                self.env.agent_view_size = self.agent_view_size
            else:
                reward = 0
                done = False
                self.done = True
        else:
            rgb_img = self.get_rgb_img(obs, self.obs_type)
        
        obs['image'] = rgb_img
        
        return obs, reward, done, {}