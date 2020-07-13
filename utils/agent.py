import numpy as np
import torch

import utils
from model import ACModel

from matplotlib import pyplot as plt

class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, env, obs_space, action_space, model_dir,
                 device=None, argmax=False, num_envs=1, use_memory=False, use_text=False):
        obs_space, self.preprocess_obs_goals = utils.get_obs_goals_preprocessor(obs_space)
        self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        status = utils.get_status(model_dir)

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size)

        self.acmodel.load_state_dict(status["model_state"])
        self.acmodel.to(self.device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obs_goals, "vocab"):
            self.preprocess_obs_goals.vocab.load_vocab(status["vocab"])
        
        self.goals = list(status['agent_goals'].values())

        for goal in self.goals:
            goal = env.unwrapped.get_obs_render(
                goal,
                tile_size=32
            )

            plt.imshow(goal)
            plt.show()

    def concat_obs_goal(self, obs):
        if 'image' in obs:
            obs_goals = [
            {
                "image": np.concatenate((obs["image"],self.goals[i]),axis=2),
                "mission": obs['mission']
            }
            for i in range(len(self.goals))]
        else:
            obs_goals = [ np.concatenate((obs,self.goals[i]),axis=2) 
                              for i in range(len(self.goals))]
        return obs_goals
    
    def get_actions(self, obss):
        actions = np.zeros(len(obss), dtype=int)

        for i in range(len(obss)):
            obs_goals = self.concat_obs_goal(obss[i])

            preprocessed_obs_goals = self.preprocess_obs_goals(obs_goals, device=self.device)

            with torch.no_grad():
                if self.acmodel.recurrent:
                    dists, values, self.memories = self.acmodel(preprocessed_obs_goals, self.memories)
                else:
                    dists, values = self.acmodel(preprocessed_obs_goals)
            print(values.data,values.data.max(0)[1])
            g = values.data.max(0)[1]
            if self.argmax:
                actions[i] = dists.probs.max(1, keepdim=True)[1][g].cpu().numpy()
            else:
                actions[i] = dists.sample()[g].cpu().numpy()
            
        return actions

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
