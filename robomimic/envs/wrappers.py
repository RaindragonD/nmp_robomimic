"""
A collection of useful environment wrappers.
"""
from copy import deepcopy
import textwrap
import gymnasium
import numpy as np
from collections import deque

import robomimic.envs.env_base as EB
import h5py
import torch

from robofin.pointcloud.torch import FrankaSampler

class EnvWrapper(gymnasium.Env):
    """
    Base class for all environment wrappers in robomimic.
    """
    def __init__(self, env):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
        """
        assert isinstance(env, EB.EnvBase) or isinstance(env, EnvWrapper)
        self.env = env

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env
        Raises:
            Exception: [Double wrapping env]
        """
        env = self.env
        while True:
            if isinstance(env, EnvWrapper):
                if env.class_name() == self.class_name():
                    raise Exception(
                        "Attempted to double wrap with Wrapper: {}".format(
                            self.__class__.__name__
                        )
                    )
                env = env.env
            else:
                break

    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment

        Returns:
            env (EnvBase instance): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    def _to_string(self):
        """
        Subclasses should override this method to print out info about the 
        wrapper (such as arguments passed to it).
        """
        return ''

    def __repr__(self):
        """Pretty print environment."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\nenv={}".format(self.env), indent)
        msg = header + '(' + msg + '\n)'
        return msg

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if id(result) == id(self.env):
                    return self
                return result

            return hooked
        else:
            return orig_attr


class FrameStackWrapper(EnvWrapper):
    """
    Wrapper for frame stacking observations during rollouts. The agent
    receives a sequence of past observations instead of a single observation
    when it calls @env.reset, @env.reset_to, or @env.step in the rollout loop.
    """
    def __init__(self, env, num_frames):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            num_frames (int): number of past observations (including current observation)
                to stack together. Must be greater than 1 (otherwise this wrapper would
                be a no-op).
        """
        assert num_frames > 1, "error: FrameStackWrapper must have num_frames > 1 but got num_frames of {}".format(num_frames)

        super(FrameStackWrapper, self).__init__(env=env)
        self.num_frames = num_frames
        # add a timestep dim to the observation space 
        for k in self.env.observation_space.spaces:
            self.env.observation_space[k] = gymnasium.spaces.Box(
                low=np.expand_dims(self.env.observation_space.spaces[k].low, axis=0).repeat(self.num_frames, axis=0),
                high=np.expand_dims(self.env.observation_space.spaces[k].high, axis=0).repeat(self.num_frames, axis=0),
                shape=(self.num_frames,) + self.env.observation_space.spaces[k].shape,
                dtype=self.env.observation_space.spaces[k].dtype,
            )
        # keep track of last @num_frames observations for each obs key
        self.obs_history = None

    def _get_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        obs_history = {}
        for k in init_obs:
            obs_history[k] = deque(
                [init_obs[k][None] for _ in range(self.num_frames)], 
                maxlen=self.num_frames,
            )
        return obs_history

    def _get_stacked_obs_from_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        return { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }

    def cache_obs_history(self):
        self.obs_history_cache = deepcopy(self.obs_history)

    def uncache_obs_history(self):
        self.obs_history = self.obs_history_cache
        self.obs_history_cache = None

    def reset(self, seed=None):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs, reset_info = self.env.reset()
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history(), reset_info

    def reset_to(self, state):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs = self.env.reset_to(state)
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def step(self, action):
        """
        Modify to update the internal frame history and return frame stacked observation,
        which will have leading dimension @self.num_frames for each key.

        Args:
            action (np.array): action to take

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, trunc, info = self.env.step(action)
        self.update_obs(obs, action=action, reset=False)
        # update frame history
        for k in obs:
            # make sure to have leading dim of 1 for easy concatenation
            self.obs_history[k].append(obs[k][None])
        obs_ret = self._get_stacked_obs_from_history()
        return obs_ret, r, done, trunc, info

    def update_obs(self, obs, action=None, reset=False):
        obs["timesteps"] = np.array([self.timestep])
        
        if reset:
            obs["actions"] = np.zeros(self.env.action_dimension)
        else:
            self.timestep += 1
            obs["actions"] = action[: self.env.action_dimension]

    def _to_string(self):
        """Info to pretty print."""
        return "num_frames={}".format(self.num_frames)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

class EvaluateOnDatasetWrapper(EnvWrapper):
    def __init__(self, env, dataset_path, filter_key="valid"):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            dataset_path (str): path to dataset
            filter_key (str): key to filter dataset on
        """
        super(EvaluateOnDatasetWrapper, self).__init__(env=env)
        self._dataset_path = dataset_path
        self._filter_key = filter_key
        self.load_evaluation_data(dataset_path)
    
    def load_evaluation_data(self, hdf5_path):
        self.hdf5_file = h5py.File(hdf5_path, "r", swmr=True, libver="latest")
        filter_key = self._filter_key
        self.demos = [
            elem.decode("utf-8")
            for elem in np.array(self.hdf5_file["mask/{}".format(filter_key)][:])
        ]
        self.initial_states = [
            dict(
                states=self.hdf5_file["data/{}/states".format(ep)][()][0],
            )
            for ep in self.demos
        ]
        try:
            self.goal_pcds = [
                self.hdf5_file["data/{}/obs/target_robot_pcd".format(ep)][()][0]
                for ep in self.demos
            ]
        except:
            self.goal_pcds = None
        
        try:
            self.actions = [
                self.hdf5_file["data/{}/actions".format(ep)][()]
                for ep in self.demos
            ]
        except:
            self.actions = None
        
        try:
            self.current_angles = [
                self.hdf5_file["data/{}/obs/current_angles".format(ep)][()]
                for ep in self.demos
            ]
        except:
            self.current_angles = None
        self.eval_index = 0
    
    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        self.timestep = 0
        if self._dataset_path is not None:
            print("Resetting to state {}".format(self.eval_index))
            state = self.initial_states[self.eval_index % len(self.initial_states)]
            self.env.reset()
            obs = self.env.reset_to(state)
            if self.goal_pcds is not None:
                goal_pcd = self.goal_pcds[self.eval_index % len(self.initial_states)]
                self.env.env.goal_pcd = goal_pcd[:, :3]
                obs['target_robot_pcd'] = goal_pcd
            self.eval_index += 1
            return obs
        else:
            return self.env.reset()
    
    def step(self, action):
        """
        Step environment.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): observation dictionary.
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        if self.actions is not None:
            correct_action = self.actions[(self.eval_index-1) % len(self.initial_states)][self.timestep]
            # mse 
            mse = np.linalg.norm(action - correct_action) ** 2
            print(f"Timestep: {self.timestep} Action error: {mse}")
        o, r, d, i = self.env.step(action)
        self.timestep += 1
        return o, r, d, i


    
class ResampleGoalPCDWrapper(EnvWrapper):
    def __init__(self, env, num_robot_points=2048):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            num_robot_points (int): number of points to sample from robot point cloud
        """
        super(ResampleGoalPCDWrapper, self).__init__(env=env)
        self.num_robot_points = num_robot_points
        self.fk_sampler = FrankaSampler("cpu", use_cache=True, num_fixed_points=4096)
    
    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        self.timestep = 0
        obs = self.env.reset()
        goal_angles = obs['goal_angles']
        self.goal_pcd = self.fk_sampler.sample(torch.tensor(goal_angles)).numpy()[0]
        self.goal_pcd = self.goal_pcd[np.random.choice(self.goal_pcd.shape[0], self.num_robot_points, replace=False)] 
        self.goal_pcd = np.concatenate([self.goal_pcd, np.zeros((self.goal_pcd.shape[0], 1)) * 2], axis=1)
        obs['goal_angles'] = self.goal_pcd
        return obs
    
    def step(self, action):
        """
        Step environment.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): observation dictionary.
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        o, r, d, i = self.env.step(action)
        o['goal_angles'] = self.goal_pcd
        self.timestep += 1
        return o, r, d, i