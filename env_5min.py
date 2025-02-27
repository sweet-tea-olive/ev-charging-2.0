import gym
from gym import spaces
import numpy as np
import random
from ChargingStations import ChargingStations
from Requests import Requests
from EVState import EVs
from data_loader import DataLoader
from utils import visualize_trajectory, print_ev_rewards_summary


class EVChargingDecisionEnv(gym.Env):
	def __init__(self, N, T, dt, delta1, delta2, discount_factor=0.99):
		super(EVChargingDecisionEnv, self).__init__()

		# Initialize environment parameters
		self.N = N  # Number of EVs
		self.T = T  # Total decision epochs
		self.dt = dt
		self.delta1 = delta1  # Charging session duration
		self.delta2 = delta2 # Charging session duration
		self.discount_factor = discount_factor

		# Action space: 2 actions (remain-idle, go-charge)
		self.action_space = spaces.MultiBinary(self.N)

		# State space: For each EV, (operational-status, time-to-next-availability, SoC, location)
		self.state_space = spaces.Dict({
			"OperationalStatus": spaces.MultiDiscrete([3] * self.N),  # 3 states: 0 (idle), 1 (serving), 2 (charging)
			"TimeToNextAvailability": spaces.MultiDiscrete([101] * self.N),  # Values from 0 to 100
			"SoC": spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),  # SoC in range [0, 1]
			"Location": spaces.Box(
				low=np.tile(np.array([0, 0]), (self.N, 1)),  # Shape becomes (self.N, 2)
				high=np.tile(np.array([101, 101]), (self.N, 1)),  # Shape becomes (self.N, 2)
				dtype=np.int32
			)
		})

	def reset(self, seed=None):

		if seed is not None:
			random.seed(seed)  # Set seed for Python's random module
			np.random.seed(seed)  # Set seed for NumPy
		

		self.current_timepoint = 0
		self.EVs = EVs(self.N)
		self.state = self.EVs.state
		self.returns = 0
		data_loader = DataLoader()
  
		arrival_rates = data_loader.load_arrival_rates("trip_data_5min_5evs2019-04.csv")
		self.requests_system = Requests(arrival_rates=arrival_rates)
		self.requests_system.load_trip_records("ready_trip_data2019-04.csv")
		self.requests_system.pay_rates = data_loader.load_pay_rates("pay_rates_5min.csv")
  
		ref_price_vector = data_loader.load_ref_price_vector("iso-2020-06-01-30min.csv")
		self.charging_system = ChargingStations(ref_price_vector)
		self.charging_system.random_register(station_id=1)

		self.dispatch_results = {i: {'order': None, 'cs': None} for i in range(self.N)}  # Initialize dispatch results

		self.agents_trajectory = {'OperationalStatus': np.zeros((self.N, self.T+1)),
								  'TimeToNextAvailability': np.zeros((self.N, self.T+1)),
								  'SoC': np.zeros((self.N, self.T+1)),
								  'Action': np.zeros((self.N, self.T), dtype=int),
								  'Reward': np.zeros((self.N, self.T))} 

		# Reset metrics
		self.total_charging_cost = 0.0
		self.total_added_soc = 0.0
		self.total_successful_dispatches = 0
		self.total_driver_pay_earned = 0.0
		self.discounted_returns = np.zeros(self.N)  # Reset discounted returns

		return np.array(self.state)

	def verify_data(self):
		plt.plot(self.requests_system.arrival_rates)
		plt.plot(self.requests_system.pay_rates)
		plt.plot(self.charging_system.ref_price_vector)

	def feasible_action(self, action):
		for i in range(self.N):
			s_i = self.state['SoC'][i]
			if s_i >= 1.0 and action[i] == 1:
				return False

			if s_i == 0.0 and action[i] == 0:
				return False

		return True

	def charge_as_needed(self, actions):
  
		for i in range(self.N):
			s_i = self.state['SoC'][i]
			if s_i >= 1 - self.EVs.reserve_SoC * 0.5 and actions[i] == 1:
				actions[i] = 0

			elif s_i <=self.EVs.reserve_SoC and actions[i] == 0:
				actions[i] = 1
    
			else:
				actions[i] = 0

		return actions

	def step(self, action):
		# action = self.charge_as_needed(action)

		num_requests = self.requests_system.arrival_rates[self.current_timepoint]*5
		self.requests_system.sample_requests(num_requests, self.current_timepoint)
		# self.charging_system.update_all_prices()

		for s in ['OperationalStatus', 'TimeToNextAvailability', 'SoC']:
			self.agents_trajectory[s][:, self.current_timepoint] = self.state[s]

		self.agents_trajectory['Action'][:, self.current_timepoint] = action

		dispatch_evs = []  
		go_charge_evs = []
		stay_charge_evs = []

		for i in range(self.N):
			o_t_i, tau_t_i, SoC_i, _ = self.EVs.get_state(i)

			# Case 1: Idle EVs taking action 0 (remain idle)
			if tau_t_i == 0 and action[i] == 0:
		
				dispatch_evs.append(i)
	 
				# If it was previously serving, remove its trip record
				if o_t_i == 1:
					self.dispatch_results[i]['order'] = None
					assert self.dispatch_results[i]['cs'] == None, "dispatch: charge records should be none"

				# If it was previously charging, release its charger slot
				if o_t_i == 2:
					charger_id = self.dispatch_results[i]['cs']['station_id']
					self.charging_system.adjust_occupancy(charger_id, 1)  # Free up a slot
					self.dispatch_results[i]['cs'] = None  # Remove charging record
					assert self.dispatch_results[i]['order'] == None, "dispatch: serving records should be none"
				
			# Case 2: Idle EVs taking action 1 (go charge)
			if tau_t_i == 0 and action[i] == 1:
				if o_t_i == 0 or o_t_i == 1:
					go_charge_evs.append(i)
    
					# in case it is serving before, remove its trip records
					self.dispatch_results[i]['order'] = None

				if o_t_i == 2:
					stay_charge_evs.append(i)
					assert self.dispatch_results[i]['order'] == None, "serving records should be none"

	  
			if tau_t_i >= 1:
				continue


		open_requests = self.requests_system.update_open_requests(self.current_timepoint)
		open_stations = self.charging_system.update_open_stations()

				
		if dispatch_evs:
			self.random_dispatch(dispatch_evs, open_requests)

		if go_charge_evs:
			self.unique_relocate(go_charge_evs, open_stations)
   
		if stay_charge_evs:
			for i in stay_charge_evs:
				SoC_i = self.EVs.get_state(i)[2]
				added_SoC = max(1. - SoC_i, 0)
	
				if added_SoC >= self.EVs.reserve_SoC * 0.5:
		
					charging_session = self.dispatch_results[i]['cs']
					charging_price = charging_session["price"] # price is locked, which means
					# can still enjoy low price if stay charge
	 
					charging_session["added_SoC"] = added_SoC
					charging_session["session_cost"] = added_SoC * self.EVs.b_cap[i] * charging_price
					charging_session["charging_duration"] = int(np.minimum(np.ceil(added_SoC/self.EVs.charge_rate[i]), self.mlp2))
					
				else:
					# forbid charging if SoC is still very high
					charger_id = self.dispatch_results[i]['cs']['station_id']
					self.charging_system.adjust_occupancy(charger_id, 1)  # Free up a slot
					self.dispatch_results[i]['cs'] = None  # Remove charging record
				assert self.dispatch_results[i]['order']==None, "stay charge: serving records should be None"

		rewards = []
		next_states = []

		for i in range(self.N):
			s_i = self.EVs.get_state(i)
			reward = self.compute_reward(i, s_i, action[i])
			next_state = self.state_transition(i, s_i, action[i])
			self.EVs.update_state(i, next_state)

			rewards.append(reward)
			next_states.append(next_state)

		for i in range(self.N):
			self.discounted_returns[i] = rewards[i] + self.discount_factor * self.discounted_returns[i]

		for i in range(self.N):
			if i in go_charge_evs and self.dispatch_results[i]['cs']:
				self.total_charging_cost += self.dispatch_results[i]['cs']['session_cost']
				self.total_added_soc += self.dispatch_results[i]['cs']['added_SoC']
	
			if i in stay_charge_evs and self.dispatch_results[i]['cs']:
				self.total_charging_cost += self.dispatch_results[i]['cs']['session_cost']
				self.total_added_soc += self.dispatch_results[i]['cs']['added_SoC']
	
			if i in dispatch_evs and self.dispatch_results[i]['order']:
				self.total_successful_dispatches += 1
				self.total_driver_pay_earned += self.dispatch_results[i]['order']['driver_pay']

		self.returns += sum(rewards)
		self.state = self.EVs.state

		self.agents_trajectory['Reward'][:, self.current_timepoint] = rewards

		self.current_timepoint += 1
  
		if self.current_timepoint >= self.T or all(self.state['SoC'][i] < self.EVs.reserve_SoC for i in range(self.N)):
			# Store final state
			for s in ['OperationalStatus', 'TimeToNextAvailability', 'SoC']:
				self.agents_trajectory[s][:, self.current_timepoint] = self.state[s]

		dones = np.array([
			self.current_timepoint >= self.T or self.state['SoC'][i] == 0.0
			for i in range(self.N)
		])
		info = {}
		for i in range(self.N):
			agent_reason = ""
			if self.current_timepoint >= self.T:
				agent_reason = "Episode End."
			elif self.state['SoC'][i] == 0.0:
				agent_reason = "Battery Depleted."
			info["agent_id"] = agent_reason
   
		# if all(dones):
		# 	# self.finalize_orders()
		# 	# self.compute_final_rewards()
			# self.render()
			# self.reset()
			# self.charging_system.sample_all_prices()

		return np.array(self.state), rewards, dones, info, {}

	def state_transition(self, ev_i, s_t_i, action_i):
		o_t_i, tau_t_i, theta_t_i, _ = s_t_i
		
		if tau_t_i == 0:  # EV is available for decision making
			if action_i == 0:  # remain-idle action
				theta_t1_i = np.maximum(theta_t_i - self.EVs.energy_consumption[ev_i]*self.dt, 0) # reduction by time step

				if self.dispatch_results[ev_i]['order']:
					o_t1_i = 1
					tau_t1_i = self.dispatch_results[ev_i]['order']['pickup_duration'] + self.dispatch_results[ev_i]['order']['trip_duration'] - 1
				else:
					o_t1_i = 0
					tau_t1_i = 0

			elif action_i == 1:  # go-charge or stay charge action
				if self.dispatch_results[ev_i]['cs']:
					o_t1_i = 2
					charging_duration = self.dispatch_results[ev_i]['cs']['charging_duration']
					tau_t1_i = np.maximum(charging_duration - self.dt, 0)

					added_SoC = self.dispatch_results[ev_i]['cs']['added_SoC']
					theta_t1_i = np.minimum(theta_t_i + added_SoC / charging_duration * self.dt, 1)

				else:
					o_t1_i = 0
					tau_t1_i = 0
					theta_t1_i = np.maximum(theta_t_i - self.EVs.energy_consumption[ev_i]*self.dt, 0)

		elif o_t_i == 1 and tau_t_i >= 1:  # EV is serving an order
			tau_t1_i = np.maximum(tau_t_i - self.dt, 0)
			theta_t1_i = np.maximum(theta_t_i - self.EVs.energy_consumption[ev_i]*self.dt, 0)
			o_t1_i = 1

		elif o_t_i == 2 and tau_t_i >= 1:  # EV is charging
			tau_t1_i = np.maximum(tau_t_i - self.dt, 0)
			charging_duration = self.dispatch_results[ev_i]['cs']['charging_duration']
			added_SoC = self.dispatch_results[ev_i]['cs']['added_SoC']
			theta_t1_i = np.minimum(theta_t_i + added_SoC / charging_duration*self.dt, 1)
			o_t1_i = 2

		x_t1_i = self.update_gps_locations(ev_i)
  
		theta_t1_i = 0.8

		return o_t1_i, tau_t1_i, theta_t1_i, x_t1_i

	def compute_reward(self, ev_i, s_t_i, action_i):
		o_t_i, tau_t_i, _, _ = s_t_i
		
		if tau_t_i == 0:
			if action_i == 0:
				if self.dispatch_results[ev_i]['order']:
					r_i = self.dispatch_results[ev_i]['order']['driver_pay']
				else:
					r_i = 0

			elif action_i == 1:
				if self.dispatch_results[ev_i]['cs']:
					r_i = -self.dispatch_results[ev_i]['cs']['session_cost']
				else:
					r_i = 0

		elif o_t_i == 1 and tau_t_i >= 1:
			r_i = 0

		elif o_t_i == 2 and tau_t_i >= 1:
			r_i = 0

		else:
			raise ValueError("Inconsistent state: EV is idle (o_t_i == 0) but has tau_t_i >= 1.")
		
		return r_i

	def render(self):
		print(f"Time step: {self.current_timepoint}, State: {self.state}")

	def report_progress(self):
		open_requests = self.requests_system.update_open_requests(self.current_timepoint*self.dt)
		open_stations = self.charging_system.update_open_stations()
		total_number_of_open = self.charging_system.get_resource_level()
  
		op_status = self.state["OperationalStatus"]
		idle_evs = [i for i, status in enumerate(op_status) if status == 0]
		serving_evs = [i for i, status in enumerate(op_status) if status == 1]
		charging_evs = [i for i, status in enumerate(op_status) if status == 2]
  
		if len(self.requests_system.requests_list) == 0:
			acceptance_rate = 0
		else:
			acceptance_rate = self.total_successful_dispatches / len(self.requests_system.requests_list)

		
		report = {
			"current_hour": self.current_timepoint//12,
			"current_minute": self.current_timepoint%12,
			"open_requests": open_requests,
			"open_charging_stations": open_stations,
			"total_number_of_open_slots": total_number_of_open,
			"EVs_idle": idle_evs,
			"EVs_serving": serving_evs,
			"EVs_charging": charging_evs,
			"total_charging_cost": self.total_charging_cost,
			"total_added_soc": self.total_added_soc,
			"total_successful_dispatches": self.total_successful_dispatches,
			"acceptance_rate": acceptance_rate,
			"total_driver_pay_earned": self.total_driver_pay_earned
		}
		print("len(self.requests_system.requests_list):", len(self.requests_system.requests_list))

		print(f"Detailed Status Report at  {self.current_timepoint//12}:{self.current_timepoint%12}:")
		print(f"  Open Requests: {open_requests}")
		print(f"  Open Charging Stations: {open_stations}")
		print(f"  Total Number of Open Slots: {total_number_of_open}")
		print(f"  Idle EVs: {idle_evs}")
		print(f"  Serving EVs: {serving_evs}")
		print(f"  Charging EVs: {charging_evs}")
		print(f"  Total Charging Cost: {self.total_charging_cost:.2f}")
		print(f"  Total Added SoC: {self.total_added_soc:.2f}")
		print(f"  Total Successful Dispatches: {self.total_successful_dispatches}")
		print(f"  Acceptance Rate: {acceptance_rate:.2f}")
		print(f"  Total driver_pay Earned: {self.total_driver_pay_earned:.2f}")
  
		return report

	def random_dispatch(self, dispatch_evs, open_requests):
		if not open_requests:
			for ev in dispatch_evs:
				self.dispatch_results[ev]['order'] = None
			return

		valid_evs = []
		for ev in dispatch_evs:
			if self.state['SoC'][ev] >= self.EVs.reserve_SoC:
				valid_evs.append(ev)
			else:
				self.dispatch_results[ev]['order'] = None

		if not valid_evs:
			return

		sorted_request_ids = sorted(open_requests, key=lambda req_id: self.requests_system.requests_list[req_id]['driver_pay'], reverse=True)

		for req_id in sorted_request_ids:
			request = self.requests_system.requests_list[req_id]
			if not valid_evs:
				break

			feasible_evs = [ev for ev in valid_evs if self.state['SoC'][ev] >= request['trip_duration'] * self.EVs.energy_consumption[ev] + self.EVs.reserve_SoC]

			if feasible_evs:
				best_ev = random.choice(feasible_evs)
				valid_evs.remove(best_ev)

				self.dispatch_results[best_ev]['order'] = {
					"driver_pay": request["driver_pay"],
					"pickup_duration": 0,
					"trip_duration": request["trip_duration"],
					"destination": (0,0)
				}
				self.requests_system.complete_request(req_id)

		for ev in valid_evs:
			self.dispatch_results[ev]['order'] = None

	def unique_relocate(self, go_charge_evs, open_stations):
		if not open_stations:
			for ev in go_charge_evs:
				self.dispatch_results[ev]['cs'] = None
			return

		for ev in go_charge_evs:
			if self.state['SoC'][ev] < self.EVs.reserve_SoC:
				self.dispatch_results[ev]['cs'] = None
				go_charge_evs.remove(ev)

		station_id = open_stations[0]
		station_info = self.charging_system.stations_list[station_id]
		number_of_open = station_info["number_of_open"]

		random.shuffle(go_charge_evs)
		evs_to_assign = go_charge_evs[:min(number_of_open, len(go_charge_evs))]
		total_assign = len(evs_to_assign)

		for ev in go_charge_evs:
			if ev in evs_to_assign:
	   
				added_SoC = max(1. - self.state["SoC"][ev], 0)
	
				if added_SoC >= self.EVs.reserve_SoC*0.5:
					self.dispatch_results[ev]['cs'] = {"station_id": station_id} 
					self.dispatch_results[ev]['cs']["added_SoC"] = added_SoC
					self.dispatch_results[ev]['cs']["charging_duration"] = int(np.minimum(np.ceil(added_SoC/self.EVs.charge_rate[ev]), self.delta1))

					charging_price = station_info["forecasted_prices"][self.current_timepoint//6]
					self.dispatch_results[ev]['cs']["price"] = charging_price

					self.dispatch_results[ev]['cs']["session_cost"] = added_SoC * self.EVs.b_cap[ev] * charging_price + station_info["one_time_fee"]
				
				else:
					self.dispatch_results[ev]['cs'] = None
					total_assign = np.maximum(total_assign-1,0)
			else:
				self.dispatch_results[ev]['cs'] = None

		self.charging_system.adjust_occupancy(station_id, -total_assign)

	def update_gps_locations(self, ev_i):
		return (0, 0)


def main():
	
	dt = 5  # 1-minute time step;  Actions taken every 5 minutes
	delta1 = 15 # charge session is 3 time steps (15 minutes)
	delta2 = 5 # charge session is 1 time steps (5 minutes)
	total_evs = 5
 
	# make sure all data in decission-interval resolution exist in the current path

	env = EVChargingDecisionEnv(total_evs, 216, dt, delta1, delta2)  # 3 EVs, total 10 sessions, charging holds 2 sessions
	
	total_episodes = 10
	ep_pay = []
	ep_cost = []
	ep_returns = []
	for ep in range(total_episodes):
		env.reset()

		for step in range(env.T):
			# action = env.action_space.sample()
			action = [0 for _ in range(env.N)]
			_, _, dones, info, _ = env.step(action)
			# print("dones:", dones)
			
			if ep % 5 == 0:
				env.report_progress()
    
			if all(dones):
				# print(info)
				# env.render()
				# if ep%10==0:
					# env.report_progress()
					# visualize_trajectory(env.agents_trajectory)
				# visualize_trajectory(env.agents_trajectory)
				# print("TimeToNextAvailability:", env.agents_trajectory["TimeToNextAvailability"])
				# print("SoC:", env.agents_trajectory["SoC"])
				# print("Action:", env.agents_trajectory["Action"])
				# print("Reward:", env.agents_trajectory['Reward'])
				# print("Total Requests:", env.requests_system.requests_list)
				# print_ev_rewards_summary(env.agents_trajectory['Reward'])
	
				break

		ep_pay.append(env.total_driver_pay_earned)
		ep_cost.append(env.total_charging_cost)
		ep_returns.append(env.returns)
  
		# if ep%10==0:

		# 	print(f"ep {ep}, returns {env.returns}.")
	visualize_trajectory(env.agents_trajectory)
	ep_pay = [round(float(r), 2) for r in ep_pay]
	print("total pay:", ep_pay)
	print("average total pay is:", sum(ep_pay)/total_episodes)	
 
	ep_cost = [round(float(r), 2) for r in ep_cost]
	print("total costs:", ep_cost)
	print("average total costs is:", sum(ep_cost)/total_episodes)	
 
	ep_returns = [round(float(r), 2) for r in ep_returns]
	print("total returns:", ep_returns)
	print("average total returns is:", sum(ep_returns)/total_episodes)	


	env.close()


if __name__ == "__main__":
	main()
 
 
# random policy, first time connection fee is 0: average total returns is: 1042.6432
# random policy, first time connection fee is 20: average total returns is:  -2036.7722 (average total costs is: 3088.3320000000003)