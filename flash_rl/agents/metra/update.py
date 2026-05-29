from dataclasses import dataclass
from typing import Any

import torch
from .agent import METRAAgent

def move_batch_to_device(batch, device: torch.device):
	return {
		key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
		for key, value in batch.items()
	}

def update_skill_rewards(state: METRAAgent, data, training_info):
	obs = data["observation"]
	next_obs = data["next_observation"]

	cur_z = state.skill_encoder(obs).mean
	next_z = state.skill_encoder(next_obs).mean
	target_z = next_z - cur_z

	rewards = (target_z * data["skill"]).sum(dim=1)

	training_info.update({
		"cur_z": cur_z,
		"next_z": next_z,
	})
	data["reward"] = rewards


def update_loss_te(state: METRAAgent, data, training_info):
	update_skill_rewards(state, data, training_info)

	rewards = data["reward"]
	obs = data["observation"]
	next_obs = data["next_observation"]

	lambda_value = state.dual_lam.param.exp()
	phi_x = training_info["cur_z"]
	phi_y = training_info["next_z"]

	cst_dist = torch.ones_like(obs[:, 0])
	square_dist = torch.square(phi_y - phi_x).mean(dim=1)
	cst_penalty = cst_dist - square_dist
	cst_penalty = torch.clamp(cst_penalty, max=state.cfg.dual_slack)

	te_obj = rewards + lambda_value.detach() * cst_penalty
	loss_te = -te_obj.mean()

	training_info.update({
		"cst_penalty": cst_penalty,
		"square_dist": square_dist,
		"loss_te": loss_te,
	})


def update_loss_dual_lam(state: METRAAgent, training_info):
	log_dual_lam = state.dual_lam.param
	dual_lam_value = log_dual_lam.exp()
	loss_dual_lam = log_dual_lam * training_info["cst_penalty"].detach().mean()
	training_info.update({
		"dual_lam": dual_lam_value,
		"loss_dual_lam": loss_dual_lam,
	})


def optimize_te(state: METRAAgent, data, training_info):
	update_loss_te(state, data, training_info)

	state.optimizers["traj_encoder"].zero_grad()
	training_info["loss_te"].backward()
	state.optimizers["traj_encoder"].step()

	update_loss_dual_lam(state, training_info)
	state.optimizers["dual_lam"].zero_grad()
	training_info["loss_dual_lam"].backward()
	state.optimizers["dual_lam"].step()


def update_loss_qf(state: METRAAgent, data, training_info):
	obs = torch.concat([data["observation"], data["skill"]], dim=-1)
	next_obs = torch.concat([data["next_observation"], data["skill"]], dim=-1)
	action = data["action"]
	rewards = data["reward"]

	with torch.no_grad():
		alpha = state.log_alpha.param.exp()

	q1_pred = state.qf1(obs, action).flatten()
	q2_pred = state.qf2(obs, action).flatten()
	next_action_dist, *_ = state.option_policy(next_obs)

	new_next_actions_pre_tanh, new_next_actions = next_action_dist.rsample_with_pre_tanh_value()
	new_next_action_log_probs = next_action_dist.log_prob(
		new_next_actions,
		pre_tanh_value=new_next_actions_pre_tanh,
	)

	target_q_values = torch.min(
		state.target_qf1(next_obs, new_next_actions).flatten(),
		state.target_qf2(next_obs, new_next_actions).flatten(),
	)
	target_q_values = target_q_values - alpha * new_next_action_log_probs
	target_q_values = target_q_values * state.cfg.discount

	with torch.no_grad():
		q_target = rewards + target_q_values
	loss_qf1 = torch.nn.functional.mse_loss(q1_pred, q_target) * 0.5
	loss_qf2 = torch.nn.functional.mse_loss(q2_pred, q_target) * 0.5

	training_info.update({
		"q_target_mean": q_target.mean(),
		"q_err_mean": ((q_target - q1_pred).mean() + (q_target - q2_pred).mean()) / 2,
		"loss_qf1": loss_qf1,
		"loss_qf2": loss_qf2,
	})


def update_loss_policy(state: METRAAgent, data, training_info):
	with torch.no_grad():
		alpha = state.log_alpha.param.exp()

	obs = torch.concat([data["observation"], data["skill"]], dim=-1)
	action_dists, *_ = state.option_policy(obs)
	new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
	new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)

	min_q_values = torch.min(
		state.qf1(obs, new_actions).flatten(),
		state.qf2(obs, new_actions).flatten(),
	)

	loss_policy = (alpha * new_action_log_probs - min_q_values).mean()
	training_info.update({
		"loss_policy": loss_policy,
		"new_action_log_probs": new_action_log_probs,
	})


def update_loss_alpha(state: METRAAgent, training_info):
	loss_alpha = (-state.log_alpha.param * (training_info["new_action_log_probs"].detach() + state.target_entropy)).mean()
	training_info.update({
		"loss_alpha": loss_alpha,
	})


def update_targets(state: METRAAgent):
	for t_param, param in zip(state.target_qf1.parameters(), state.qf1.parameters()):
		t_param.data.copy_(t_param.data * (1 - state.cfg.tau) + param.data * state.cfg.tau)

	for t_param, param in zip(state.target_qf2.parameters(), state.qf2.parameters()):
		t_param.data.copy_(t_param.data * (1 - state.cfg.tau) + param.data * state.cfg.tau)


def optimize_op(state: METRAAgent, data, training_info):
	update_loss_qf(state, data, training_info)
	state.optimizers["qf"].zero_grad()
	(training_info["loss_qf1"] + training_info["loss_qf2"]).backward()
	state.optimizers["qf"].step()

	update_loss_policy(state, data, training_info)
	state.optimizers["option_policy"].zero_grad()
	training_info["loss_policy"].backward()
	state.optimizers["option_policy"].step()

	update_loss_alpha(state, training_info)
	state.optimizers["log_alpha"].zero_grad()
	training_info["loss_alpha"].backward()
	state.optimizers["log_alpha"].step()

	update_targets(state)


def train_once(state: METRAAgent):
	data = move_batch_to_device(state.replay_buffer.sample(), state.device)
	training_info = {}

	optimize_te(state, data, training_info)
	update_skill_rewards(state, data, training_info)
	optimize_op(state, data, training_info)

	return training_info, data


__all__ = [
	"NotebookMETRAState",
	"move_batch_to_device",
	"train_once",
]
