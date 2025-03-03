"""
This module implements the EV Charging environment training and evaluation
pipeline, including the definition of Q-network, Whittle Index Network, and
the WIQLearningAgent. It also contains functions for training, evaluation,
plotting, and saving/loading model configurations.
"""

import os
import json
import random
from collections import deque

import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from ChargingStations import ChargingStations
from TripRequests import TripRequests
from EVFleet import EVFleet
from EVChargingEnv import EVChargingEnv
from utils import df_to_list, load_file, visualize_trajectory, print_ev_rewards_summary, plot_scores


# =============================================================================
# Q-Network and Whittle Index Network Definitions
# =============================================================================
class QNetwork(nn.Module):
	"""
	Simple feed-forward Q-network that takes state and lambda value as input.
	"""
	def __init__(self, state_size, action_size):
		super(QNetwork, self).__init__()
		# Adding 1 to state_size to account for lambda_g
		self.fc1 = nn.Linear(state_size + 1, 100)
		self.fc2 = nn.Linear(100, 100)
		self.fc3 = nn.Linear(100, action_size)

	def forward(self, x, lambda_value):
		# Concatenate the state and lambda value
		x = torch.cat([x, lambda_value], dim=1)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)


class WhittleIndexNetwork(nn.Module):
	"""
	Network to compute Whittle indices from state.
	"""
	def __init__(self, state_size, action_size):
		super(WhittleIndexNetwork, self).__init__()
		self.fc1 = nn.Linear(state_size, 100)
		self.fc2 = nn.Linear(100, 100)
		# Output dimension is action_size - 1 as per design
		self.fc3 = nn.Linear(100, action_size - 1)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)


# =============================================================================
# WIQLearningAgent Definition
# =============================================================================
class WIQLearningAgent:
	"""
	Agent that combines Q-learning with Whittle index approximation for resource allocation.
	"""
	def __init__(self, state_size, action_size, gamma=0.99, lr_q=3e-4, lr_w=3e-4,
				 epsilon=1, epsilon_decay=0.995, tau=1e-3, q_update_frequency=1,
				 w_update_frequency=10, alpha_lambda=0.01, max_resource_level=10,
				 feature_scaling=False):
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = gamma
		self.lr_q = lr_q
		self.lr_w = lr_w
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.tau = tau
		self.q_update_frequency = q_update_frequency
		self.w_update_frequency = w_update_frequency
		self.alpha_lambda = alpha_lambda
		self.feature_scaling = feature_scaling

		# Networks
		self.qnetwork_local = QNetwork(state_size, action_size)
		self.qnetwork_target = QNetwork(state_size, action_size)
		self.whittle_index_network = WhittleIndexNetwork(state_size, action_size)
		self.optimizer_q = optim.Adam(self.qnetwork_local.parameters(), lr=lr_q)
		self.optimizer_w = optim.Adam(self.whittle_index_network.parameters(), lr=lr_w)

		# Experience replay memory
		self.memory = deque(maxlen=10000)
		self.batch_size = 64

		# Time step counters for updates
		self.q_t_step = 0
		self.w_t_step = 0

		# Global cost (lambda) table for resource levels
		self.lambda_table = {i: 0.0 for i in range(max_resource_level + 1)}
		self.global_cost = 0.01

		# For feature scaling of states
		self.state_min = np.array([0, 0, 0, 0, 0])
		self.state_max = np.array([2, 100, 1, 0, 0])

	def update_state_bounds(self, state):
		"""Update the minimum and maximum observed state values."""
		self.state_min = np.minimum(self.state_min, state)
		self.state_max = np.maximum(self.state_max, state)

	def normalize_state(self, state):
		"""Normalize state based on observed state bounds."""
		if self.feature_scaling:
			state = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)
		return state

	def select_action(self, state, epsilon=None):
		"""
		Select an action using an ε-greedy policy.
		If epsilon is provided (e.g., epsilon=0 during evaluation), it overrides self.epsilon.
		"""
		if epsilon is None:
			epsilon = self.epsilon

		if random.random() < epsilon:
			return random.choice(range(self.action_size))  # Exploration

		self.update_state_bounds(state)
		state = np.array(self.normalize_state(state))
		state_tensor = torch.from_numpy(state).float().unsqueeze(0)
		lambda_g = torch.tensor([[self.global_cost]], dtype=torch.float)

		with torch.no_grad():
			q_values = self.qnetwork_local(state_tensor, lambda_g)
		return np.argmax(q_values.numpy())  # Exploitation

	def store_transition(self, state, action, reward, next_state, done):
		"""Store a transition tuple in replay memory."""
		self.memory.append((state, action, reward, next_state, done))

	def update_q_values(self):
		"""Sample a batch from memory and update the Q-network."""
		if len(self.memory) < self.batch_size:
			return
		batch = random.sample(self.memory, self.batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)
		for s in states:
			self.update_state_bounds(s)
		states = np.array([self.normalize_state(s) for s in states], dtype=np.float32)
		actions = torch.tensor(actions, dtype=torch.long)
		rewards = torch.tensor(rewards, dtype=torch.float)
		for s in next_states:
			self.update_state_bounds(s)
		next_states = np.array([self.normalize_state(s) for s in next_states], dtype=np.float32)
		dones = torch.tensor(dones, dtype=torch.float)

		states = torch.from_numpy(states)
		next_states = torch.from_numpy(next_states)
		lambda_tensors = torch.tensor([[self.global_cost]] * self.batch_size, dtype=torch.float)
		q_values = self.qnetwork_local(states, lambda_tensors).gather(1, actions.unsqueeze(1)).squeeze()
		next_q_values = self.qnetwork_target(next_states, lambda_tensors).max(1)[0]

		# Adjust rewards with global cost
		adjusted_rewards = rewards - (actions.float() * self.global_cost)
		targets = adjusted_rewards + (self.gamma * next_q_values * (1 - dones))

		loss = nn.MSELoss()(q_values, targets)
		self.optimizer_q.zero_grad()
		loss.backward()
		self.optimizer_q.step()

		self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

	def update_whittle_indices(self):
		"""Update the Whittle index network based on sampled transitions."""
		if len(self.memory) < self.batch_size:
			return
		batch = random.sample(self.memory, self.batch_size)
		states = np.array([self.normalize_state(e[0]) for e in batch], dtype=np.float32)
		for s in states:
			self.update_state_bounds(s)
		states = torch.from_numpy(states)

		whittle_indices = self.whittle_index_network(states)
		# Reshape to match lambda dimension
		lambda_tensors = whittle_indices.squeeze(1).unsqueeze(1)
		q_values = self.qnetwork_local(states, lambda_tensors).detach()
		q_1 = q_values[:, 1]
		q_0 = q_values[:, 0]
		wi_updates = q_1 - q_0

		# Loss for Whittle network update
		loss = nn.MSELoss()(whittle_indices.squeeze(1), whittle_indices.squeeze(1) + wi_updates)
		self.optimizer_w.zero_grad()
		loss.backward()
		self.optimizer_w.step()

	def update_lambda_table(self, resource_usage, resource_level):
		"""Update the global cost (lambda) for a given resource level."""
		self.lambda_table[resource_level] = max(
			0, self.lambda_table[resource_level] + self.alpha_lambda * (resource_usage - resource_level)
		)

	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters."""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

	def decay_epsilon(self):
		"""Decay the exploration rate."""
		self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

	def step(self, state, action, reward, next_state, done):
		"""
		Process a transition: store it, update Q-values and Whittle indices, and decay epsilon.
		"""
		self.store_transition(state, action, reward, next_state, done)
		self.q_t_step = (self.q_t_step + 1) % self.q_update_frequency
		self.w_t_step = (self.w_t_step + 1) % self.w_update_frequency
		if self.q_t_step == 0:
			self.update_q_values()
		if self.w_t_step == 0:
			self.update_whittle_indices()
		self.decay_epsilon()

	def save(self, qnetwork_path, whittle_index_network_path, lambda_table_path):
		"""Save model parameters and lambda table."""
		torch.save(self.qnetwork_local.state_dict(), qnetwork_path)
		torch.save(self.whittle_index_network.state_dict(), whittle_index_network_path)
		with open(lambda_table_path, 'w') as f:
			json.dump(self.lambda_table, f)

	def load(self, qnetwork_path, whittle_index_network_path, lambda_table_path):
		"""Load model parameters and lambda table."""
		self.qnetwork_local.load_state_dict(torch.load(qnetwork_path))
		self.qnetwork_target.load_state_dict(torch.load(qnetwork_path))
		self.whittle_index_network.load_state_dict(torch.load(whittle_index_network_path))
		with open(lambda_table_path, 'r') as f:
			self.lambda_table = json.load(f)
		self.lambda_table = {int(k): v for k, v in self.lambda_table.items()}


# =============================================================================
# Training, Evaluation, and Plotting Functions
# =============================================================================
def train(env, agent, n_episodes=200, max_t=3000, epsilon_start=1.0,
		  epsilon_end=0.01, epsilon_decay=0.995):
	"""
	Train the agent over a given number of episodes.
	
	Returns:
		ep_rewards (list): Total reward for each episode.
		epsilons (list): Epsilon value for each episode.
	"""
	ep_rewards = []
	for i_episode in range(1, n_episodes + 1):
		env.reset()
		ep_reward = 0
		for t in range(max_t):
			states = [np.hstack(env.evs.get_state(ev)) for ev in range(env.N)]
			resource_level = env.charging_stations.get_resource_level()
			agent.global_cost = agent.lambda_table[resource_level]

			actions = [agent.select_action(states[ev]) for ev in range(env.N)]
			# actions = [action.item() if isinstance(action, np.int64) else action for action in actions]
			# print("states:", states)
			# print("actions:", actions)

			# Environment step with multi-agent actions
			_, rewards, done, _, violation_penalty = env.step(actions)
			next_states = [np.hstack(env.evs.get_state(ev)) for ev in range(env.N)]
			resource_usage = np.sum(actions)
			agent.update_lambda_table(resource_usage, resource_level)
   
			num_agents_with_action_1 = sum(1 for action in actions if action == 1)

			if num_agents_with_action_1 > 0:
				avg_resource_penalty = violation_penalty["resource_constraint"] / num_agents_with_action_1
			else:
				avg_resource_penalty = 0.0  # Avoid division by zero

			# Update each agent's experience
			for ev in range(env.N):
				agent.step(states[ev], actions[ev], rewards[ev]-violation_penalty[ev]-avg_resource_penalty, next_states[ev], done)
				states[ev] = next_states[ev]

			if i_episode % 50 == 0:
				env.report_progress()

			if done:
				break

			ep_reward += sum(rewards)

		ep_rewards.append(ep_reward)
		print(f"Episode {i_episode}\tTotal Reward: {ep_reward:.2f}")

	epsilons = [max(epsilon_end, epsilon_decay**i) for i in range(n_episodes)]
	return ep_rewards, epsilons


def evaluate(env, agent, n_episodes=10, max_t=3000):
	"""
	Evaluate the trained agent over multiple episodes.
	
	Returns:
		ep_rewards (list): Total reward for each evaluation episode.
	"""
	ep_rewards = []
	for i_episode in range(1, n_episodes + 1):
		# For evaluation, we typically want deterministic behavior (epsilon=0)
		env.reset()
		
		total_reward = 0
		for t in range(max_t):
			states = [np.hstack(env.evs.get_state(ev)) for ev in range(env.N)]
			# Use epsilon=0 for greedy action selection during evaluation
			actions = [agent.select_action(states[ev], epsilon=0) for ev in range(env.N)]
			
			_, rewards, done, _, _ = env.step(actions)
			total_reward += sum(rewards)
			if done:
				break

		ep_rewards.append(total_reward)
		print(f"Evaluation Episode {i_episode}\tTotal Reward: {total_reward:.2f}")

	avg_reward = np.mean(ep_rewards)
	std_reward = np.std(ep_rewards)
	print(f"\nEvaluation over {n_episodes} episodes:")
	print(f"Mean Reward: {avg_reward:.2f}, Std Dev: {std_reward:.2f}")
	return sum(ep_rewards)/len(ep_rewards)


def plot_rewards(ep_rewards, epsilons, window=100):
	"""
	Plot raw and smoothed rewards as well as epsilon decay over training episodes.
	"""
	smoothed_rewards = np.convolve(ep_rewards, np.ones(window) / window, mode='valid')
	plt.figure(figsize=(10, 5))
	plt.plot(ep_rewards, label="Raw Rewards", alpha=0.3)
	plt.plot(smoothed_rewards, label=f"Moving Average (window={window})", color="red")
	plt.xlabel("Episode")
	plt.ylabel("Total Reward")
	plt.title("Training Convergence")
	plt.legend()
	plt.show()

	plt.figure(figsize=(10, 5))
	plt.plot(epsilons)
	plt.xlabel("Episode")
	plt.ylabel("Epsilon")
	plt.title("Epsilon Decay Over Time")
	plt.show()


def compute_agent_state_size(state_space, num_agents):
	"""
	Compute the per-agent state size based on the environment's state space.
	
	For a MultiDiscrete space, assume one element per agent.
	For a Box space, divide the total number of elements by the number of agents.
	"""
	size = 0
	for key, space in state_space.spaces.items():
		if isinstance(space, spaces.MultiDiscrete):
			size += 1
		elif isinstance(space, spaces.Box):
			size += int(np.prod(space.shape) // num_agents)
		else:
			raise NotImplementedError(f"Unsupported space type: {type(space)}")
	return size


# =============================================================================
# Main Function: Training and Evaluation Pipeline
# =============================================================================
def train_model():
	# Load training configuration
	with open("config_for_train.json", "r") as config_file:
		train_config = json.load(config_file)
	env = EVChargingEnv(train_config, training_mode=True)
	env.reset()

	state_size = compute_agent_state_size(env.state_space, env.N)
	action_size = 2
	max_resource_level = env.charging_stations.get_resource_level()

	# Initialize the agent for training
	agent = WIQLearningAgent(state_size, action_size, feature_scaling=True, max_resource_level=max_resource_level)

	# Train the agent and plot rewards
	ep_rewards, epsilons = train(env, agent, n_episodes=20)
	plot_rewards(ep_rewards, epsilons)
	visualize_trajectory(env.agents_trajectory)
	print_ev_rewards_summary(env.agents_trajectory['Reward'])
	print("lambda_table:", agent.lambda_table)

	# Save the trained model and lambda table
	output_dir = "wi_output"
	os.makedirs(output_dir, exist_ok=True)
	agent.save(os.path.join(output_dir, "qnetwork_local.pth"),
			   os.path.join(output_dir, "whittle_index_network.pth"),
			   os.path.join(output_dir, "lambda_table.json"))
	
	return agent

def eval_model(agent):
	
	# Evaluate the agent: for each evaluation config, create a new environment instance.
	# Here, we show one evaluation config as an example.
	# output_dir = "wi_output"
	# agent = WIQLearningAgent(state_size, action_size, feature_scaling=False)
	# agent.load(os.path.join(output_dir, "qnetwork_local.pth"),
	#            os.path.join(output_dir, "whittle_index_network.pth"),
	#            os.path.join(output_dir, "lambda_table.json"))

	eval_rewards = []
	total_eval_eps = 1  # Adjust as needed for evaluation
	for ep in range(1, 1+total_eval_eps):
		config_filename = f"config_for_eval_{ep}.json"
		with open(config_filename, "r") as config_file:
			eval_config = json.load(config_file)
		env = EVChargingEnv(eval_config, training_mode=False)
		env.reset()
		# print(env.trip_requests.requests)
		

		ep_reward = evaluate(env, agent, n_episodes=1)
		eval_rewards.append(ep_reward)
	
	avg_total_reward = sum(eval_rewards) / len(eval_rewards)
	print("Average total rewards:", avg_total_reward)
	visualize_trajectory(env.agents_trajectory)
	print_ev_rewards_summary(env.agents_trajectory['Reward'])

def main():
	agent = train_model()
	eval_model(agent)
	
if __name__ == "__main__":
	main()
