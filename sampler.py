import torch
import numpy as np

def gen_one_skill(skill_dim: int = 2):
    skills = torch.randn((1, skill_dim), device=torch.device('cpu'))
    return torch.nn.functional.normalize(skills, dim=-1).numpy()

import gymnasium as gym
class Sampler:
    def __init__(self, env: gym.Env):
        self.env = env
        self.skill = None

    def _update_skill(self):
        self.skill = gen_one_skill()

    def rollout(self, policy, path_len: int = 200, training=True):
        """
        required_keys = {
            "observation",
            "action",
            "reward",
            "terminated",
            "truncated",
            "next_observation",
            "skill",
            "skill_resample_step",
        }
        """

        observations = []
        actions = []
        rewards = []
        next_observations = []
        

        prev_obs, _ = self.env.reset()
        self._update_skill()
        
        for i in range(path_len):
            skill_obs = np.concatenate([prev_obs, self.skill], axis=-1)

            if (training):
                action, agent_info = policy.get_sample_actions(skill_obs)
            else:
                action, agent_info = policy.get_mode_actions(skill_obs)
                
            action = np.squeeze(action, axis=0)

            next_obs, rewd, trun, done, _  = self.env.step(action)

            if trun or done:
                raise NotImplementedError

            observations.append(prev_obs)
            actions.append(action)
            rewards.append(rewd)
            next_observations.append(next_obs)

            prev_obs = next_obs

        observations = np.array(observations)
        actions = np.array(actions)
        next_observations = np.array(next_observations)

        return {
            "observation": torch.squeeze(torch.Tensor(observations)),
            "action": torch.squeeze(torch.Tensor(actions)),
            "reward": torch.Tensor(rewards),
            "terminated": torch.zeros(size=(path_len,), dtype=bool),
            "truncated": torch.zeros(size=(path_len,), dtype=bool),
            "next_observation": torch.squeeze(torch.Tensor(next_observations)),
            "skill": torch.from_numpy(np.repeat(self.skill, repeats=path_len, axis=0)),
            "skill_resample_step": torch.zeros(size=(path_len,))
        }