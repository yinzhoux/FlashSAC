import os

import tqdm
from flash_rl.agents.metra.agent import METRAAgent
from gymnasium import Env

from utils import sample_one_path, sample_from_buffer, fix_batch_train_agent, plot_fix_batch_history
from plot_metra_eval import generate_eval_artifacts
from utils import evaluate

def train(agent: METRAAgent, env: Env, cfg, n_epoch: int = 50, eval_interval: int = 10):
    # reset
    agent._replay_buffer.reset()
    device = agent._device
    os.makedirs("./pngs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./evals", exist_ok=True)

    for epoch_idx in range(n_epoch):
        print("Epoch:", epoch_idx, "....")

        # Sampling
        for _ in range(8):
            sample_one_path(env, agent, cfg, obs_norm=False)
        
        # Updating
        historys = []
        for update_idx in tqdm.tqdm(range(50)):
            
            # batch["observation"] = self._observations[idxs]
            # batch["action"] = self._actions[idxs]
            # batch["reward"] = self._rewards[idxs]
            # batch["terminated"] = self._terminateds[idxs]
            # batch["truncated"] = self._truncateds[idxs]
            # batch["next_observation"] = self._next_observations[idxs]
            # batch["skill"] = self._skills[idxs]
            # batch["skill_resample_step"] = self._skill_resample_steps[idxs]
        
            batch = sample_from_buffer(agent._replay_buffer, device, return_full_batch=True)

            


        # Plot
        if epoch_idx % eval_interval == 0:
            plot_fix_batch_history(historys, show=False, path=f'./pngs/step{epoch_idx}.png')
            agent.save(f'./models/step{epoch_idx}')
            generate_eval_artifacts(agent, env, cfg, f'./evals/step{epoch_idx}')