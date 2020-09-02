from abc import ABC, abstractmethod
import numpy as np
import torch
import random
import hashlib

from torch_ac.format import default_preprocess_obs_goals
from torch_ac.utils import dictlist, penv
from matplotlib import pyplot as plt

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obs_goals, reshape_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obs_goals : function
            a function that takes observations returned by the environment
            with the goals the agent was trying to reach and converts them 
            into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = penv.ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obs_goals = preprocess_obs_goals or default_preprocess_obs_goals
        self.reshape_reward = reshape_reward
        self.eps = 0.1
        self.her = 2
        self.N = 0

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize obs and goals

        self.obs = self.env.reset()

        self.goals = {}
        if 'image' in self.obs[0]:
            self.goals[hash(str(self.obs[0]['image']*0))] = self.obs[0]['image']*0
        else:
            self.goals[hash(str(self.obs[0]*0))] = self.obs[0]*0

        goals = list(self.goals.values())
        self.goal = np.array([random.sample(goals,1)[0] for _ in range(self.num_procs)])

        self.obs_goal = None
        
        # Initialize experience values

        shape = (self.num_frames_per_proc*self.her, self.num_procs)
        self.exps = [None]*(shape[0])
        self.obs_goals = [None]*(shape[0])
        self.dones = [0]*(shape[0])
        
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def to_hash(self,obs):
        hash_obs = hashlib.md5(obs.tostring()).hexdigest()
        return hash_obs
        
    def update_goal(self, processes=None, strategy='best'):
        # Update goal with epsilon greedy generalised policy improvement

        goals = list(self.goals.values())
        
        processes = processes if processes else range(self.num_procs)
        for i in processes:
            if strategy == 'random':
                self.goal[i] = random.sample(goals,1)[0]
                continue

            obs = self.obs[i]
            memory = self.memory[i]
            
            if 'image' in obs:
                obs_goals = [
                {
                    "image": np.concatenate((obs["image"],goals[i]),axis=2),
                    "mission": obs['mission']
                }
                for i in range(len(goals))]
            else:
                obs_goals = [ np.concatenate((obs,goals[i]),axis=2) 
                                for i in range(len(goals))]
            
            preprocessed_obs_goals = self.preprocess_obs_goals(obs_goals, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    memory = torch.stack([memory]*len(goals),0)
                    dists, values, _ = self.acmodel(preprocessed_obs_goals, memory)
                else:
                    dists, values = self.acmodel(preprocessed_obs_goals)
            best_goal_idx = values.data.max(0)[1]

            self.goal[i] = goals[best_goal_idx]
    
        
    def concat_obs_goal(self):
        """
        TODO
        """
            
        if 'image' in self.obs[0]:
            self.obs_goal = [
            {
                "image": np.concatenate((self.obs[i]["image"],self.goal[i]),axis=2),
                "mission": self.obs[i]['mission']
            }
            for i in range(self.num_procs)]
        else:
            self.obs_goal = [
            {
                "image": np.concatenate((self.obs[i],self.goal[i]),axis=2)
            }
            for i in range(self.num_procs)]

    def extended_reward(self, obs, goal, action, reward, done,b=False):   
        """
        TODO
        """

        if 'image' in obs:
            obs = obs['image']
        if self.to_hash(obs) != self.to_hash(goal) and done:  
            reward = self.N
            
        return reward

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            self.update_goal()
            self.concat_obs_goal()
            preprocessed_obs_goal = self.preprocess_obs_goals(self.obs_goal, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs_goal, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs_goal)
            action = dist.sample()
            obs, reward, done, _ = self.env.step(action.cpu().numpy())
            
            # Update experiences values

            self.exps[i] = (self.obs, action, reward, done, obs)
            self.obs_goals[i] = self.obs_goal
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.dones[i] = done
            self.actions[i] = action
            self.values[i] = value
            extended_reward = [
                self.extended_reward(obs_, goal_, action_, reward_, done_)
                for obs_, goal_, action_, reward_, done_ in zip(self.obs, self.goal, action, reward, done)
            ]
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, goal_, action_, reward_, done_)
                    for obs_, goal_, action_, reward_, done_ in zip(self.obs, self.goal, action, extended_reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(extended_reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update goals
            
            for j, done_ in enumerate(done):
                if done_:
                    if 'image' in self.obs[j]:
                        self.goals[self.to_hash(self.obs[j]['image'])] = self.obs[j]['image']
                    else:
                        self.goals[self.to_hash(self.obs[j])] = self.obs[j]
                    # self.update_goal(processes=[j])
            self.obs = obs
            
            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Hindsight experience replay
        
        for i in range(self.num_frames_per_proc, len(self.exps)):
            e = i-self.num_frames_per_proc
            self.obs, action, reward, done, obs = self.exps[e]

            b=False
            dones = np.array(self.dones[e:self.num_frames_per_proc])+0
            for p in range(self.num_procs):
                next_goals = np.where(dones[:,p]==1)[0]+e
                if len(next_goals) != 0 and next_goals[0] >= e:
                    if e==next_goals[0]:
                        b=True
                    obs_ = self.exps[next_goals[0]][0][p]
                    if 'image' in obs_:
                        obs_ = obs_['image']
                    self.goal[p] = obs_
                # else:
                #     goals = list(self.goals.values())
                #     self.goal[p] = random.sample(goals,1)[0]
            
            self.concat_obs_goal()
            preprocessed_obs_goal = self.preprocess_obs_goals(self.obs_goal, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs_goal, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs_goal)
            
            # Update experiences values

            self.obs_goals[i] = self.obs_goal
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            extended_reward = [
                self.extended_reward(obs_, goal_, action_, reward_, done_,b=b)
                for obs_, goal_, action_, reward_, done_ in zip(self.obs, self.goal, action, reward, done)
            ]
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, goal_, action_, reward_, done_)
                    for obs_, goal_, action_, reward_, done_ in zip(self.obs, self.goal, action, extended_reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(extended_reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            self.obs = obs


        # Add advantage and return to experiences

        preprocessed_obs_goal = self.preprocess_obs_goals(self.obs_goal, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs_goal, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs_goal)

        for i in reversed(range(len(self.exps))):
            next_mask = self.masks[i+1] if i < len(self.exps) - 1 else self.mask
            next_value = self.values[i+1] if i < len(self.exps) - 1 else next_value
            next_advantage = self.advantages[i+1] if i < len(self.exps) - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is len(self.exps),
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = dictlist.DictList()
        exps.obs_goal = [self.obs_goals[i][j]
                    for j in range(self.num_procs)
                    for i in range(len(self.exps))]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs_goal = self.preprocess_obs_goals(exps.obs_goal, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
