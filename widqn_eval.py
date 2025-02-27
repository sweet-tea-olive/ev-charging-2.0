import gym
import torch
import json
import numpy as np
import pandas as pd
import random
from torch import nn
from env_5min import EVChargingDecisionEnv
from widqn_greedy import compute_agent_state_size, WIQLearningAgent
from utils import visualize_trajectory, print_ev_rewards_summary, plot_scores

# def select_action(agent, state): 
# 	state = agent.normalize_state(state)
# 	state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
# 	with torch.no_grad():
# 		whittle_indices = agent.whittle_index_network(state_tensor).cpu().numpy().flatten()
# 	print("whittle_indices:", whittle_indices)
# 	print("agent.global_cost:", agent.global_cost)
# 	action = 1 if whittle_indices[0] > agent.global_cost+0.01 else 0
# 	return action, whittle_indices[0]

def select_action(agent, state): 
	state = agent.normalize_state(state)
	state_tensor = torch.from_numpy(state).float().unsqueeze(0)
	lambda_g = torch.tensor([[agent.global_cost]], dtype=torch.float)
	with torch.no_grad():
		q_values = agent.qnetwork_local(state_tensor, lambda_g)

	return np.argmax(q_values.numpy())
	
def run_inference(env, agent, n_episodes=10, max_t=300):
	ep_rewards = []

	for i_episode in range(1, n_episodes + 1):
		env.reset()
		episode_rewards = np.zeros(env.N)
		states = [np.hstack(env.EVs.get_state(ev)) for ev in range(env.N)]
		ep_reward = 0
		for t in range(max_t):
				
			resource_level = env.charging_system.get_resource_level()
			agent.global_cost = agent.lambda_table[resource_level]
			
			actions = []
			for ev in range(env.N):
				action = select_action(agent, states[ev])
				actions.append(action)

			_, rewards, dones, _, _ = env.step(actions)
			next_states = [np.hstack(env.EVs.get_state(ev)) for ev in range(env.N)]
			
			ep_reward += sum(rewards)

			for ev in range(env.N):
				episode_rewards[ev] += rewards[ev]
				states[ev] = next_states[ev]
				
			if i_episode % 50 == 0:
				env.report_progress()


			if all(dones):
				break

		ep_rewards.append(ep_reward)
		print(f"Episode {i_episode}\t Mean Reward: {ep_reward:.2f}")

	return ep_rewards

def main():
	dt = 5  # 1-minute time step;  Actions taken every 5 minutes
	delta1 = 15 # charge session is 3 time steps (15 minutes)
	delta2 = 5 # charge session is 1 time steps (5 minutes)
	total_evs = 5
 
	# make sure all data in decission-interval resolution exist in the current path

	env = EVChargingDecisionEnv(total_evs, 216, dt, delta1, delta2)  # 3 EVs, total 10 sessions, charging holds 2 sessions
 
	state_size = compute_agent_state_size(env.state_space, env.N)
	action_size = 2
	agent = WIQLearningAgent(state_size, action_size, feature_scaling=False)
	agent.load("qnetwork_local.pth", "whittle_index_network.pth", "lambda_table.json")
	
	ep_rewards = run_inference(env, agent)
	print("average total rewards:", sum(ep_rewards)/len(ep_rewards))
	visualize_trajectory(env.agents_trajectory)
	print_ev_rewards_summary(env.agents_trajectory['Reward'])


if __name__ == "__main__":
	main()