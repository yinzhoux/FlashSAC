import gymnasium as gym
import numpy as np
import torch

def gen_one_skill(skill_dim: int = 2) -> np.ndarray:
    skills = torch.randn((1, skill_dim), device=torch.device("cpu"))
    return torch.nn.functional.normalize(skills, dim=-1).numpy()


class Sampler:
    def __init__(self, env: gym.Env, skill_dim: int, skill_sampler=gen_one_skill):
        self.env = env
        self.skill_dim = skill_dim
        self.skill_sampler = skill_sampler
        self.skill = None

    def _update_skill(self):
        self.skill = self.skill_sampler(self.skill_dim)

    def rollout(self, policy, path_len: int = 200, training: bool = True):
        observations = []
        actions = []
        rewards = []
        next_observations = []

        prev_obs, _ = self.env.reset()
        self._update_skill()

        for _ in range(path_len):
            skill_obs = np.concatenate([prev_obs, self.skill], axis=-1)

            if training:
                action, _agent_info = policy.get_sample_actions(skill_obs)
            else:
                action, _agent_info = policy.get_mode_actions(skill_obs)

            action = np.squeeze(action, axis=0)
            next_obs, reward, truncated, done, _ = self.env.step(action)

            if truncated or done:
                raise NotImplementedError

            observations.append(prev_obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            prev_obs = next_obs

        observations = np.array(observations)
        actions = np.array(actions)
        next_observations = np.array(next_observations)

        return {
            "observation": torch.squeeze(torch.tensor(observations, dtype=torch.float32)),
            "action": torch.squeeze(torch.tensor(actions, dtype=torch.float32)),
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "terminated": torch.zeros(size=(path_len,), dtype=bool),
            "truncated": torch.zeros(size=(path_len,), dtype=bool),
            "next_observation": torch.squeeze(torch.tensor(next_observations, dtype=torch.float32)),
            "skill": torch.from_numpy(np.repeat(self.skill, repeats=path_len, axis=0)),
            "skill_resample_step": torch.zeros(size=(path_len,)),
        }
