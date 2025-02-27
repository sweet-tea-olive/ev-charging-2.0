import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
import random
from collections import deque
from torch import nn, optim
from ChargingStations import ChargingStations
from Requests import Requests
from EVState import EVs
from utils import visualize_trajectory, print_ev_rewards_summary, plot_scores
from env_5min import EVChargingDecisionEnv


# Define the Actor network
class ActorNetwork(nn.Module):
	def __init__(self, state_size, action_size):
		super(ActorNetwork, self).__init__()
		self.fc1 = nn.Linear(state_size, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, action_size)
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
		return torch.softmax(self.fc3(x), dim=-1)

# Define the Critic network
class CriticNetwork(nn.Module):
	def __init__(self, state_size):
		super(CriticNetwork, self).__init__()
		self.fc1 = nn.Linear(state_size, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 1)
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

class PPOAgent:
	def __init__(self, state_size, action_size, seed=0, lr=3e-4, gamma=0.98, clip_epsilon=0.2, update_timestep=2000, k_epochs=4, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, feature_scaling=False):
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)

		self.actor = ActorNetwork(state_size, action_size)
		self.critic = CriticNetwork(state_size)
		self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

		self.gamma = gamma
		self.clip_epsilon = clip_epsilon
		self.update_timestep = update_timestep
		self.k_epochs = k_epochs

		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min

		self.memory = []
		self.timestep = 0
  
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
	
	def select_action(self, state):
		
		self.update_state_bounds(state)  # Update state bounds with new state
		state = np.array(self.normalize_state(state))
		state = torch.from_numpy(state).float().unsqueeze(0)

		if random.random() < self.epsilon:
			action = np.random.choice(self.action_size)
			log_prob = torch.log(torch.tensor(1 / self.action_size))
		else:
			with torch.no_grad():
				probs = self.actor(state)
			action = np.random.choice(self.action_size, p=probs.numpy().flatten())
			log_prob = torch.log(probs.squeeze(0)[action])
		return action, log_prob

	def store_transition(self, state, action, log_prob, reward, next_state, done):
		self.memory.append((state, action, log_prob, reward, next_state, done))
		self.timestep += 1

	def update(self):
		states, actions, log_probs, rewards, next_states, dones = zip(*self.memory)
		for s in states:
			self.update_state_bounds(s)  # Update state bounds with batch of states
		states = np.array([self.normalize_state(s) for s in states], dtype=np.float32)
		actions = torch.tensor(actions, dtype=torch.long)
		log_probs = torch.tensor(log_probs, dtype=torch.float)
		rewards = torch.tensor(rewards, dtype=torch.float)
		for s in next_states:
			self.update_state_bounds(s)  # Update state bounds with batch of next states
		next_states = np.array([self.normalize_state(s) for s in next_states], dtype=np.float32)
		dones = torch.tensor(dones, dtype=torch.float)
  
		states = torch.from_numpy(states)
		next_states = torch.from_numpy(next_states)

		returns = []
		discounted_sum = 0
		for reward, done in zip(reversed(rewards), reversed(dones)):
			if done:
				discounted_sum = 0
			discounted_sum = reward + (self.gamma * discounted_sum)
			returns.insert(0, discounted_sum)
		returns = torch.tensor(returns, dtype=torch.float)

		for _ in range(self.k_epochs):
			state_values = self.critic(states).squeeze()
			advantages = returns - state_values.detach()

			new_log_probs = torch.log(self.actor(states).gather(1, actions.unsqueeze(1)).squeeze())
			ratios = torch.exp(new_log_probs - log_probs.detach())

			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
			actor_loss = -torch.min(surr1, surr2).mean()

			critic_loss = nn.MSELoss()(state_values, returns)

			loss = actor_loss + 0.5 * critic_loss
			self.optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
			nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
			self.optimizer.step()

		self.memory = []
		self.timestep = 0
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
		
	def load(self, path):
		self.actor.load_state_dict(torch.load(path))
		
	def save(self, path):
		torch.save(self.actor.state_dict(), path)

# Training function with PPO and moving average calculation
def train(env, agent, n_episodes=200, max_t=3000, window_size=100):
	scores = []
	scores_window = deque(maxlen=window_size)  # Last 100 scores for moving average
	ep_rewards = []

	for i_episode in range(1, n_episodes + 1):
		env.reset()
		episode_rewards = np.zeros(env.N)
		
		ep_reward = 0
		states = [np.hstack(env.EVs.get_state(ev)) for ev in range(env.N)]

		for t in range(max_t):
			
			actions = []
			log_probs = []

			for ev in range(env.N):
				action, log_prob = agent.select_action(states[ev])
				actions.append(action)
				log_probs.append(log_prob)
				
			# actions = env.preprocess_action(actions)
			
			_, rewards, dones, _, _ = env.step(actions)
			next_states = [np.hstack(env.EVs.get_state(ev)) for ev in range(env.N)]

			for ev in range(env.N):
				
				agent.store_transition(states[ev], actions[ev], log_probs[ev], rewards[ev], next_states[ev], dones[ev])
				# episode_rewards[ev] += rewards[ev] * (agent.gamma ** t)
				episode_rewards[ev] += rewards[ev]
				states[ev] = next_states[ev]
	
			ep_reward += sum(rewards)

			if i_episode % 50 == 0:
				env.report_progress()

			if all(dones) or agent.timestep >= agent.update_timestep:
				agent.update()

			if all(dones):
				break

		ep_rewards.append(ep_reward)


		episode_mean_reward = np.mean(episode_rewards)
		scores.append(episode_mean_reward)
		scores_window.append(episode_mean_reward)

		# Log intermediate scores for debugging
		print(f"Episode {i_episode}\t Mean Reward: {ep_reward:.2f}")

		# if len(scores_window) == window_size:
		# 	moving_average = np.mean(scores_window)
			# print(f"Episode {i_episode}\tMoving Average: {moving_average:.2f}")

	return scores

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
 
	state_size = compute_agent_state_size(env.state_space, env.N)
	action_size = 2
	agent = PPOAgent(state_size, action_size, seed=0,feature_scaling=True)
	
	scores = train(env, agent)
	plot_scores(scores)
	visualize_trajectory(env.agents_trajectory)
	print_ev_rewards_summary(env.agents_trajectory['Reward'])
	
	# Save the trained model
	agent.save("ppo_agent.pth")

if __name__ == "__main__":
	main()