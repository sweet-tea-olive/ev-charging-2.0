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
import json
import os


# Define the Q-network
class QNetwork(nn.Module):
	def __init__(self, state_size, action_size):
		super(QNetwork, self).__init__()
		self.fc1 = nn.Linear(state_size + 1, 100)  # state_size + 1 to account for lambda_g
		self.fc2 = nn.Linear(100, 100)
		self.fc3 = nn.Linear(100, action_size)
		# self.init_weights()

	def init_weights(self):
		nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
		nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
		nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
		nn.init.constant_(self.fc1.bias, 0.01)
		nn.init.constant_(self.fc2.bias, 0.01)
		nn.init.constant_(self.fc3.bias, 0.01)

	def forward(self, x, lambda_value):
		x = torch.cat([x, lambda_value], dim=1)  # Concatenate state and lambda_value
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)

# Define the Whittle index network
class WhittleIndexNetwork(nn.Module):
	def __init__(self, state_size, action_size):
		super(WhittleIndexNetwork, self).__init__()
		self.fc1 = nn.Linear(state_size, 100)
		self.fc2 = nn.Linear(100, 100)
		self.fc3 = nn.Linear(100, action_size - 1)  # action_size - 1
		# self.init_weights()

	def init_weights(self):
		nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
		nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
		nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
		nn.init.constant_(self.fc1.bias, 0.01)
		nn.init.constant_(self.fc2.bias, 0.01)
		nn.init.constant_(self.fc3.bias, 0.01)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)

class WIQLearningAgent:
	def __init__(self, state_size, action_size, gamma=0.99, lr_q=3e-4, lr_w=3e-4, epsilon=1, epsilon_decay=0.995, tau=1e-3, q_update_frequency=1, w_update_frequency=10, alpha_lambda=0.01, max_resource_level=10, feature_scaling=False):
		self.state_size = state_size
		self.action_size = action_size
		self.gamma = gamma
		self.lr_q = lr_q
		self.lr_w = lr_w
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.qnetwork_local = QNetwork(state_size, action_size)
		self.qnetwork_target = QNetwork(state_size, action_size)
		self.whittle_index_network = WhittleIndexNetwork(state_size, action_size)
		self.optimizer_q = optim.Adam(self.qnetwork_local.parameters(), lr=lr_q)
		self.optimizer_w = optim.Adam(self.whittle_index_network.parameters(), lr=lr_w)
		self.memory = deque(maxlen=10000)
		self.batch_size = 64
		self.tau = tau
		self.q_update_frequency = q_update_frequency
		self.w_update_frequency = w_update_frequency
		self.q_t_step = 0
		self.w_t_step = 0
		self.alpha_lambda = alpha_lambda
		self.lambda_table = {i: 0.0 for i in range(max_resource_level+1)}  # Tabular storage for global costs
		self.global_cost = 0.01
		self.feature_scaling = feature_scaling
		self.state_min = np.array([0,0,0,0,0])
		self.state_max = np.array([2,100,1,0,0])

	def update_state_bounds(self, state):
		self.state_min = np.minimum(self.state_min, state)
		self.state_max = np.maximum(self.state_max, state)

	def normalize_state(self, state):
		if self.feature_scaling:
			state = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)  # Add small epsilon to avoid division by zero
		return state
		
	def select_action(self, state): # select actions without constraints

		if random.random() < self.epsilon:
			return random.choice(range(self.action_size))

		self.update_state_bounds(state)  # Update state bounds with new state
		state = np.array(self.normalize_state(state))
		state_tensor = torch.from_numpy(state).float().unsqueeze(0)
		lambda_g = torch.tensor([[self.global_cost]], dtype=torch.float)
		with torch.no_grad():
			q_values = self.qnetwork_local(state_tensor, lambda_g)

		return np.argmax(q_values.numpy())

	def store_transition(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))


	def update_q_values(self):
		if len(self.memory) < self.batch_size:
			return
		batch = random.sample(self.memory, self.batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)
		for s in states:
			self.update_state_bounds(s)  # Update state bounds with batch of states
		states = np.array([self.normalize_state(s) for s in states], dtype=np.float32)
		actions = torch.tensor(actions, dtype=torch.long)
		rewards = torch.tensor(rewards, dtype=torch.float)
		for s in next_states:
			self.update_state_bounds(s)  # Update state bounds with batch of next states
		next_states = np.array([self.normalize_state(s) for s in next_states], dtype=np.float32)
		dones = torch.tensor(dones, dtype=torch.float)

		states = torch.from_numpy(states)
		next_states = torch.from_numpy(next_states)

		lambda_tensors = torch.tensor([[self.global_cost]] * self.batch_size, dtype=torch.float)
		q_values = self.qnetwork_local(states, lambda_tensors).gather(1, actions.unsqueeze(1)).squeeze()
		next_q_values = self.qnetwork_target(next_states, lambda_tensors).max(1)[0]
		
		# Adjust rewards with lambda
		adjusted_rewards = rewards - (actions.float() * self.global_cost)
		
		# Compute targets using target network
		targets = adjusted_rewards + (self.gamma * next_q_values * (1 - dones))

		# Compute loss and update local Q-network
		loss = nn.MSELoss()(q_values, targets)
		self.optimizer_q.zero_grad()
		loss.backward()
		self.optimizer_q.step()

		# Soft update target network
		self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

	def update_whittle_indices(self):
		if len(self.memory) < self.batch_size:
			return
		batch = random.sample(self.memory, self.batch_size)
		states = np.array([self.normalize_state(e[0]) for e in batch], dtype=np.float32)  # Apply normalization here
		for s in states:
			self.update_state_bounds(s)  # Update state bounds with batch of states
		states = torch.from_numpy(states)

		# Update Whittle index network
		whittle_indices = self.whittle_index_network(states)
		lambda_tensors = whittle_indices.squeeze(1).unsqueeze(1)
		q_values = self.qnetwork_local(states, lambda_tensors).detach()
		q_1 = q_values[:, 1]
		q_0 = q_values[:, 0]
		
		whittle_indices = whittle_indices.squeeze(1)
		wi_updates = q_1 - q_0
		
		loss = nn.MSELoss()(whittle_indices, whittle_indices + wi_updates)
		self.optimizer_w.zero_grad()
		loss.backward()
		self.optimizer_w.step()
		
	def update_lambda_table(self, resource_usage, resource_level):
		self.lambda_table[resource_level] = max(0, self.lambda_table[resource_level] + self.alpha_lambda * (resource_usage - resource_level))

	def soft_update(self, local_model, target_model, tau):
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

	def decay_epsilon(self):
		self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)

	def step(self, state, action, reward, next_state, done):
		self.store_transition(state, action, reward, next_state, done)
		self.q_t_step = (self.q_t_step + 1) % self.q_update_frequency
		self.w_t_step = (self.w_t_step + 1) % self.w_update_frequency
		if self.q_t_step == 0:
			self.update_q_values()
		if self.w_t_step == 0:
			self.update_whittle_indices()
		self.decay_epsilon()
		
	def save(self, qnetwork_path, whittle_index_network_path, lambda_table_path):
		torch.save(self.qnetwork_local.state_dict(), qnetwork_path)
		torch.save(self.whittle_index_network.state_dict(), whittle_index_network_path)
		with open(lambda_table_path, 'w') as f:
			json.dump(self.lambda_table, f)

	def load(self, qnetwork_path, whittle_index_network_path, lambda_table_path):
		self.qnetwork_local.load_state_dict(torch.load(qnetwork_path))
		self.qnetwork_target.load_state_dict(torch.load(qnetwork_path))
		self.whittle_index_network.load_state_dict(torch.load(whittle_index_network_path))
		with open(lambda_table_path, 'r') as f:
			self.lambda_table = json.load(f)
		self.lambda_table = {int(k): v for k, v in self.lambda_table.items()}
			
# Training function with total reward calculation
def train(env, agent, n_episodes=200, max_t=3000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
	ep_rewards = []
	for i_episode in range(1, n_episodes + 1):
		env.reset()

		episode_rewards = np.zeros(env.N)  # Track rewards for each agent
		states = [np.hstack(env.EVs.get_state(ev)) for ev in range(env.N)]  # Initial states for all agents
		ep_reward = 0
		for t in range(max_t):
			
			resource_level = env.charging_system.get_resource_level()  # Get available resources
			# print("resource_level:", resource_level)
			agent.global_cost = agent.lambda_table[resource_level]
			
			actions = []
			for ev in range(env.N):
				action = agent.select_action(states[ev])
				actions.append(action)
			

			# print("actions:", actions)
				
			# Step using actions for all agents
			_, rewards, dones, _, _ = env.step(actions)
			next_states = [np.hstack(env.EVs.get_state(ev)) for ev in range(env.N)]
			
			resource_usage = np.sum(actions)  # Calculate total resource usage
			agent.update_lambda_table(resource_usage, resource_level)
			# Update agent for each EV
			for ev in range(env.N):
				
				agent.step(states[ev], actions[ev], rewards[ev], next_states[ev], dones[ev])
				episode_rewards[ev] += rewards[ev]
				states[ev] = next_states[ev]

			if i_episode % 50 == 0:
				env.report_progress()

			if all(dones):
				break

			ep_reward += sum(rewards)

		ep_rewards.append(ep_reward)

		# epsilon = max(epsilon_end, epsilon_decay * epsilon)
		
		# Log total reward for each episode
		print(f"Episode {i_episode}\tTotal Reward: {ep_reward:.2f}")
		
	return ep_rewards

def compute_agent_state_size(state_space, num_agents):
	"""_example_
	  - For a MultiDiscrete space defined as MultiDiscrete([3]*num_agents):
		  which means there are num_agents element, and each element an take values from 0 up to 2 (since 3 means there are 3 possible values: 0, 1, 2).
		  So each agent has one element → add 1.
	  - For a Box space with shape (num_agents,):
		  np.prod(space.shape) = N, and per-agent size = N/N = 1.
	  - For a Box space with shape (2, num_agents):
		  np.prod(space.shape) = 2 * N, and per-agent size = (2*N)/N = 2.
	"""
	size = 0
	for key, space in state_space.spaces.items():
		if isinstance(space, spaces.MultiDiscrete):
			# Each EV gets one element from each MultiDiscrete array
			size += 1
		elif isinstance(space, spaces.Box):
			# For per-agent state, we divide by the number of agents
			size += int(np.prod(space.shape) // num_agents)
		else:
			raise NotImplementedError(f"Unsupported space type: {type(space)}")
	return size


def main():
	dt = 5  # 1-minute time step;  Actions taken every 5 minutes
	delta1 = 15 # charge session is 3 time steps (15 minutes)
	delta2 = 5 # charge session is 1 time steps (5 minutes)
	total_evs = 5
 
	# make sure all data in decission-interval resolution exist in the current path

	env = EVChargingDecisionEnv(total_evs, 216, dt, delta1, delta2)  # 3 EVs, total 10 sessions, charging holds 2 sessions

	env.reset()
	state_size = compute_agent_state_size(env.state_space, env.N)
	action_size = 2
	max_resource_level = env.charging_system.get_resource_level() 
	agent = WIQLearningAgent(state_size, action_size, feature_scaling=True, max_resource_level=max_resource_level)
	
	scores = train(env, agent)
	# plot_scores(scores)
	visualize_trajectory(env.agents_trajectory)
	print_ev_rewards_summary(env.agents_trajectory['Reward'])
	print("lambda_table:", agent.lambda_table )
	
	
	agent.save("qnetwork_local.pth", "whittle_index_network.pth", "lambda_table.json")
 
	
	# output_dir = "wi_output"
	# os.makedirs(output_dir, exist_ok=True)
	
	# agent.save(os.path.join(output_dir, "qnetwork_local.pth"),
	# 		   os.path.join(output_dir, "whittle_index_network.pth"),
	# 		   os.path.join(output_dir, "lambda_table.json"))

	# trajectory_fig_path = os.path.join(output_dir, "trajectory_plot.png")
	# visualize_trajectory(env.agents_trajectory, save_path=trajectory_fig_path)


if __name__ == "__main__":
	main()
