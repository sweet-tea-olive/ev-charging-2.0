import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
import random
from collections import deque
from torch import nn, optim
from ChargingStations import ChargingStations
from TripRequests import Requests
from EVFleet import EVs
from utils import visualize_trajectory, print_ev_rewards_summary, plot_scores
from env_5min import EVChargingDecisionEnv
from ppo_greedy import PPOAgent, compute_agent_state_size

def run_inference(env, agent, n_episodes=200, max_t=300):

	ep_rewards = []
	for i_episode in range(1, n_episodes + 1):
		env.reset()
		
		states = [np.hstack(env.EVs.get_state(ev)) for ev in range(env.N)]
		ep_reward = 0
		for t in range(max_t):
				
			actions = []

			for ev in range(env.N):
				action, _ = agent.select_action(states[ev])
				actions.append(action)
			
			_, rewards, dones, _, _ = env.step(actions)
			next_states = [np.hstack(env.EVs.get_state(ev)) for ev in range(env.N)]

			for ev in range(env.N):
				
				states[ev] = next_states[ev]

			ep_reward += sum(rewards)

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
	# agent = PPOAgent(state_size, action_size)
	agent = PPOAgent(state_size, action_size, seed=0, feature_scaling=True)

	# Load the trained model
	agent.load("ppo_agent.pth")
	
	# Load trip records from CSV
	# trip_records = load_trip_records('trip_records_eval.csv')

	# Run inference
	ep_rewards = run_inference(env, agent)
	print("average total rewards:", sum(ep_rewards)/len(ep_rewards))
	visualize_trajectory(env.agents_trajectory)
	print_ev_rewards_summary(env.agents_trajectory['Reward'])

if __name__ == "__main__":
	main()
	