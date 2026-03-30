"""
Atari environment wrapper with standardized preprocessing
"""
import gymnasium as gym
import numpy as np
from typing import Tuple, Optional
import cv2

# Import wrappers properly
try:
    from gymnasium.wrappers import AtariPreprocessing, FrameStack, RecordEpisodeStatistics
except ImportError:
    from gymnasium.wrappers import RecordEpisodeStatistics
    # Fallback for older gymnasium versions
    AtariPreprocessing = None
    FrameStack = None


class AtariEnvWrapper:
    """Wrapper for Atari environments with RL-standard preprocessing"""
    
    def __init__(
        self,
        env_name: str = "PongNoFrameskip-v4",
        frame_size: Tuple[int, int] = (84, 84),
        frame_stack: int = 4,
        seed: Optional[int] = None,
        grayscale: bool = False,
        vit_mode: bool = True,
    ):
        self.env_name = env_name
        self.vit_mode = vit_mode
        
        # Create base environment
        self.env = gym.make(env_name, render_mode=None)
        
        # Set seed
        if seed is not None:
            self.env.reset(seed=seed)
            np.random.seed(seed)
        
        # Apply preprocessing
        if vit_mode:
            self.env = ViTAtariPreprocessing(
                self.env,
                frame_size=(224, 224),
                grayscale_obs=False,
            )
        else:
            if AtariPreprocessing:
                self.env = AtariPreprocessing(
                    self.env,
                    frame_skip=1,
                    screen_size=frame_size[0],
                    grayscale_obs=grayscale,
                    scale_obs=True,
                )
            else:
                # Manual preprocessing fallback
                self.env = ViTAtariPreprocessing(
                    self.env,
                    frame_size=frame_size,
                    grayscale_obs=grayscale,
                )
        
        # Frame stacking (optional, using custom implementation if needed)
        if frame_stack > 1 and FrameStack:
            self.env = FrameStack(self.env, num_stack=frame_stack)
        
        # Episode statistics
        self.env = RecordEpisodeStatistics(self.env)
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        self.env.close()
    
    @property
    def unwrapped(self):
        return self.env.unwrapped


class ViTAtariPreprocessing(gym.ObservationWrapper):
    """Custom preprocessing for ViT-based agents"""
    
    def __init__(
        self,
        env: gym.Env,
        frame_size: Tuple[int, int] = (224, 224),
        grayscale_obs: bool = False,
        normalize: bool = True,
    ):
        super().__init__(env)
        self.frame_size = frame_size
        self.grayscale_obs = grayscale_obs
        self.normalize = normalize
        
        if grayscale_obs:
            obs_shape = (*frame_size, 1)
        else:
            obs_shape = (*frame_size, 3)
        
        self.observation_space = gym.spaces.Box(
            low=0 if not normalize else 0.0,
            high=255 if not normalize else 1.0,
            shape=obs_shape,
            dtype=np.float32 if normalize else np.uint8,
        )
    
    def observation(self, obs):
        obs = cv2.resize(obs, self.frame_size, interpolation=cv2.INTER_AREA)
        
        if self.grayscale_obs and len(obs.shape) == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=-1)
        
        if self.normalize:
            obs = obs.astype(np.float32) / 255.0
        
        return obs


def make_parallel_envs(
    env_name: str,
    num_envs: int = 8,
    seed: int = 42,
    vit_mode: bool = True,
):
    """Create vectorized parallel environments"""
    from gymnasium.vector import AsyncVectorEnv
    import gymnasium as gym
    
    # Check if it's an Atari environment
    is_atari = "NoFrameskip" in env_name or "ALE" in env_name
    
    def make_env(rank: int):
        def _init():
            if is_atari:
                env = AtariEnvWrapper(
                    env_name=env_name,
                    seed=seed + rank,
                    vit_mode=vit_mode,
                )
                return env.env
            else:
                # For non-Atari envs (like CartPole), create simple gym env
                env = gym.make(env_name)
                env.reset(seed=seed + rank)
                # Wrap to add preprocessing if needed
                if vit_mode:
                    env = SimpleViTWrapper(env)
                return env
        return _init
    
    envs = AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    return envs


class SimpleViTWrapper(gym.ObservationWrapper):
    """Simple wrapper for non-visual environments to work with ViT pipeline"""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Keep original observation space for non-visual envs
        # The policy will handle it appropriately
    
    def observation(self, obs):
        return obs
