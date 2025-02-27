import gym
from gym import spaces
import numpy as np
import random
from ChargingStations import ChargingStations
from TripRequests import TripRequests
from EVFleet import EVFleet
from data_loader import DataLoader
import matplotlib.pyplot as plt
from utils import df_to_list, visualize_trajectory, print_ev_rewards_summary


class EVChargingEnv(gym.Env):
	def __init__(self, config):
		"""
		Initializes the EV charging environment using a single config dictionary.
		
		:param config: Dictionary containing all necessary initialization parameters.
		"""
		self.T = config.get("total_time_steps", 1)
		self.dt = config.get("time_step_minutes", 5)
		self.N = config.get("total_evs", 5)
		self.delta1 = config.get("committed_charging_block_minutes", 15)
		self.delta2 = config.get("renewed_charging_block_minutes", 5)
	   
		self.evs = EVFleet(config["ev_params"])
		self.trip_requests = TripRequests(config["trip_params"])
		self.charging_stations = ChargingStations(config["charging_params"])
		
		self.other_env_params = config.get("env_params", {})
  
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
		
		self._setup_environment()
	
	def _setup_environment(self):
			
		self.randomize_trips = True # by default, randomly sample trip requests
		self.randomize_init_SoCs = True # by default, randomize initial SoCs
		self.randomize_prices = False # by default, not randomize charging price

		self.states_min = np.array([0,0,0,0,0])
		self.states_max = np.array([2,100,1,0,0])
  

	def reset(self, seed=None):
		if seed is not None:
			random.seed(seed)  
			np.random.seed(seed) 
   
		self.evs.reset(randomize_init_SoCs=self.randomize_init_SoCs)
		self.trip_requests.reset(randomize_trips=self.randomize_trips)
		self.charging_stations.reset(randomize_prices=self.randomize_prices)
  
		self.current_timepoint = 0
		self.states = self.evs.get_all_states()
		
		self.dispatch_results = {i: {'order': None, 'cs': None} for i in range(self.N)}  # Initialize dispatch results
		self.agents_trajectory = {'OperationalStatus': np.zeros((self.N, self.T+1)),
								  'TimeToNextAvailability': np.zeros((self.N, self.T+1)),
								  'SoC': np.zeros((self.N, self.T+1)),
								  'Action': np.zeros((self.N, self.T), dtype=int),
								  'Reward': np.zeros((self.N, self.T))} 

		# Reset metrics
		self.total_charging_cost = 0.0
		self.total_added_soc = 0.0
		self.total_trip_requests = 0
		self.total_successful_dispatches = 0
		self.total_driver_pay_earned = 0.0
		self.total_violation_penalty = 0.0
		self.ep_returns = 0 # accumulate total returns over T times steps for each episode

		return np.array(self.states)

	def show_config(self):
		# print(self.trip_requests.arrival_rates)
		# print(self.trip_requests.pay_rates)
		# print(self.charging_stations.stations[1]["real_time_prices"])
		plt.plot(self.trip_requests.arrival_rates, marker='o')
		plt.plot(self.trip_requests.pay_rates, marker='x')
		plt.plot(self.charging_stations.stations[1]["real_time_prices"], marker='s')
		plt.show()
  
	def check_action_feasibility(self, actions):
		violation_penalty = 0
		for i in range(self.N):
			o_t_i, tau_t_i, SoC_i, _ = self.evs.get_state(i)

			# case 1: is serving or charging but take action 1
			if tau_t_i >= 1 and actions[i] == 1:
				violation_penalty += 100
			# case 2: already very high SoC but take action 1
			if SoC_i >= 1.0 - self.evs.reserve_SoC * 0.5 and actions[i] == 1:
				violation_penalty += 100
			# case 3: already very low SoC but take action 0
			if SoC_i <= self.evs.reserve_SoC and actions[i] == 0:
				violation_penalty += 100

		# case 4: exceed available chargers
		if sum(actions) > self.charging_stations.get_resource_level():
			violation_penalty += 100

		return violation_penalty
  
		# or raise exception
		# if violation_penalty > 0:
		# 	raise Exception("Infeasible action.")

	def step(self, actions):
		violation_penalty = self.check_action_feasibility(actions)
		self.total_violation_penalty += violation_penalty
  
		if self.randomize_trips:
			num_requests = self.trip_requests.arrival_rates[self.current_timepoint] * 5
			self.trip_requests.sample_requests(num_requests, self.current_timepoint)

		for s in ['OperationalStatus', 'TimeToNextAvailability', 'SoC']:
			self.agents_trajectory[s][:, self.current_timepoint] = self.states[s]

		self.agents_trajectory['Action'][:, self.current_timepoint] = actions

		dispatch_evs = []  
		go_charge_evs = []
		stay_charge_evs = []

		for i in range(self.N):
			o_t_i, tau_t_i, SoC_i, _ = self.evs.get_state(i)

			# Case 1: Idle EVs taking action 0 (remain idle)
			if tau_t_i == 0 and actions[i] == 0:
	
				dispatch_evs.append(i)
	 
				# If it was previously serving, remove its trip record
				if o_t_i == 1:
					self.dispatch_results[i]['order'] = None
					assert self.dispatch_results[i]['cs'] == None, "to be dispatch: previous charge records should be none"

				# If it was previously charging, release its charger slot
				if o_t_i == 2:
					charger_id = self.dispatch_results[i]['cs']['station_id']
					self.charging_stations.adjust_occupancy(charger_id, 1)  # Free up a slot
					self.dispatch_results[i]['cs'] = None  # Remove charging record
					assert self.dispatch_results[i]['order'] == None, "to be dispatch: previous serving records should be none"
				
			# Case 2: Idle EVs taking action 1 (go charge)
			if tau_t_i == 0 and actions[i] == 1:
				if o_t_i == 0 or o_t_i == 1:
					go_charge_evs.append(i)
	
					# in case it is serving before, remove its trip records
					self.dispatch_results[i]['order'] = None

				if o_t_i == 2:
					stay_charge_evs.append(i)
					assert self.dispatch_results[i]['order'] == None, "to relocate to charge: previous serving records should be none"

	  
			if tau_t_i >= 1:
				continue


		open_requests = self.trip_requests.update_open_requests(self.current_timepoint)
		open_stations = self.charging_stations.update_open_stations()

				
		if dispatch_evs:
			self.random_dispatch(dispatch_evs, open_requests)

		if go_charge_evs:
			self.unique_relocate(go_charge_evs, open_stations)
   
		if stay_charge_evs:
			for i in stay_charge_evs:
				SoC_i = self.evs.get_state(i)[2]
				added_SoC = max(1. - SoC_i, 0)
	
				if added_SoC >= self.evs.reserve_SoC * 0.5:
		
					charging_session = self.dispatch_results[i]['cs']
					charging_price = charging_session["price"] # price is locked, which means
					# can still enjoy low price if stay charge
	 
					charging_session["added_SoC"] = added_SoC
					charging_session["session_cost"] = added_SoC * self.evs.b_cap[i] * charging_price
					charging_session["charging_duration"] = int(np.minimum(np.ceil(added_SoC/self.evs.charge_rate[i]), self.delta2))
					
				else:
					# forbid charging if SoC is still very high
					charger_id = self.dispatch_results[i]['cs']['station_id']
					self.charging_stations.adjust_occupancy(charger_id, 1)  # Free up a slot
					self.dispatch_results[i]['cs'] = None  # Remove charging record
				
				assert self.dispatch_results[i]['order']==None, "to stay charge: serving records should be None"

		rewards = []
		next_states = []

		for i in range(self.N):
			s_i = self.evs.get_state(i)
			reward = self.compute_reward(i, s_i, actions[i])
			next_state = self.state_transition(i, s_i, actions[i])
			self.evs.update_state(i, next_state)

			rewards.append(reward)
			next_states.append(next_state)

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

		self.ep_returns += sum(rewards)
		self.states = self.evs.get_all_states()

		self.agents_trajectory['Reward'][:, self.current_timepoint] = rewards

		self.current_timepoint += 1

		done = False
		if self.current_timepoint >= self.T or all(self.states['SoC'][i] == 0.0 for i in range(self.N)):
			done = True
   			# Store final state
			for s in ['OperationalStatus', 'TimeToNextAvailability', 'SoC']:
				self.agents_trajectory[s][:, self.current_timepoint] = self.states[s]

		info = {}
		
		if self.current_timepoint >= self.T:
			info = "Episode End."
		if all(self.states['SoC'][i] == 0.0 for i in range(self.N)):
			info = "All Battery Depleted."

		return np.array(self.states), rewards, done, info, {"violation_penalty":violation_penalty}

	def state_transition(self, ev_i, s_t_i, action_i):
		o_t_i, tau_t_i, theta_t_i, _ = s_t_i
		
		if tau_t_i == 0:  # EV is available for decision making
			if action_i == 0:  # remain-idle action
				theta_t1_i = np.maximum(theta_t_i - self.evs.energy_consumption[ev_i]*self.dt, 0) # reduction by time step

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
					theta_t1_i = np.maximum(theta_t_i - self.evs.energy_consumption[ev_i]*self.dt, 0)

		elif o_t_i == 1 and tau_t_i >= 1:  # EV is serving an order
			tau_t1_i = np.maximum(tau_t_i - self.dt, 0)
			theta_t1_i = np.maximum(theta_t_i - self.evs.energy_consumption[ev_i]*self.dt, 0)
			o_t1_i = 1

		elif o_t_i == 2 and tau_t_i >= 1:  # EV is charging
			tau_t1_i = np.maximum(tau_t_i - self.dt, 0)
			charging_duration = self.dispatch_results[ev_i]['cs']['charging_duration']
			added_SoC = self.dispatch_results[ev_i]['cs']['added_SoC']
			theta_t1_i = np.minimum(theta_t_i + added_SoC / charging_duration*self.dt, 1)
			o_t1_i = 2

		x_t1_i = self.update_gps_locations(ev_i)

		# theta_t1_i = 0.8

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
		print(f"Time step: {self.current_timepoint}, State: {self.states}")
  
  
	def report_progress(self):
		open_requests = self.trip_requests.update_open_requests(self.current_timepoint*self.dt)
		open_stations = self.charging_stations.update_open_stations()
		total_available_chargers = self.charging_stations.get_resource_level()
  
		op_status = self.states["OperationalStatus"]
		idle_evs = [i for i, status in enumerate(op_status) if status == 0]
		serving_evs = [i for i, status in enumerate(op_status) if status == 1]
		charging_evs = [i for i, status in enumerate(op_status) if status == 2]
  
		if len(self.trip_requests.requests) == 0:
			acceptance_rate = 0
		else:
			acceptance_rate = self.total_successful_dispatches / len(self.trip_requests.requests)

		
		report = {
			"current_hour": (6*60+self.current_timepoint*self.dt)//60,
			"current_minute": (6*60+self.current_timepoint*self.dt)%60,
			"open_requests": open_requests,
			"open_charging_stations": open_stations,
			"total_available_chargers_slots": total_available_chargers,
			"EVs_idle": idle_evs,
			"EVs_serving": serving_evs,
			"EVs_charging": charging_evs,
			"total_charging_cost": self.total_charging_cost,
			"total_added_soc": self.total_added_soc,
			"total_successful_dispatches": self.total_successful_dispatches,
			"acceptance_rate": acceptance_rate,
			"total_driver_pay_earned": self.total_driver_pay_earned,
			"total_violation_penalty": self.total_violation_penalty,
		}
		print("len(self.trip_requests.requests_list):", len(self.trip_requests.requests))

		print(f"Detailed Status Report at  {(6*60+self.current_timepoint*self.dt)//60}:{(6*60+self.current_timepoint*self.dt)%60}:")
		print(f"  Open Requests: {open_requests}")
		print(f"  Open Charging Stations: {open_stations}")
		print(f"  Total Number of Open Slots: {total_available_chargers}")
		print(f"  Idle EVs: {idle_evs}")
		print(f"  Serving EVs: {serving_evs}")
		print(f"  Charging EVs: {charging_evs}")
		print(f"  Total Charging Cost: {self.total_charging_cost:.2f}")
		print(f"  Total Added SoC: {self.total_added_soc:.2f}")
		print(f"  Total Successful Dispatches: {self.total_successful_dispatches}")
		print(f"  Acceptance Rate: {acceptance_rate:.2f}")
		print(f"  Total driver_pay Earned: {self.total_driver_pay_earned:.2f}")
		print(f"  Total Violation Penalty: {self.total_violation_penalty:.2f} ")
  
		return report

	def random_dispatch(self, dispatch_evs, open_requests):
		if not open_requests:
			for ev in dispatch_evs:
				self.dispatch_results[ev]['order'] = None
			return

		valid_evs = []
		for ev in dispatch_evs:
			if self.states['SoC'][ev] >= self.evs.reserve_SoC:
				valid_evs.append(ev)
			else:
				self.dispatch_results[ev]['order'] = None

		if not valid_evs:
			return

		sorted_request_ids = sorted(open_requests, key=lambda req_id: self.trip_requests.requests[req_id]['driver_pay'], reverse=True)

		for req_id in sorted_request_ids:
			request = self.trip_requests.requests[req_id]
			if not valid_evs:
				break

			feasible_evs = [ev for ev in valid_evs if self.states['SoC'][ev] >= request['trip_duration'] * self.evs.energy_consumption[ev] + self.evs.reserve_SoC]

			if feasible_evs:
				best_ev = random.choice(feasible_evs)
				valid_evs.remove(best_ev)

				self.dispatch_results[best_ev]['order'] = {
					"driver_pay": request["driver_pay"],
					"pickup_duration": 0,
					"trip_duration": request["trip_duration"],
					"destination": (0,0)
				}
				self.trip_requests.complete_request(req_id)

		for ev in valid_evs:
			self.dispatch_results[ev]['order'] = None

	def unique_relocate(self, go_charge_evs, open_stations):
		if not open_stations:
			for ev in go_charge_evs:
				self.dispatch_results[ev]['cs'] = None
			return

		if len(go_charge_evs)==0:
			return

		for ev in go_charge_evs:
			if self.states['SoC'][ev] < self.evs.reserve_SoC:
				self.dispatch_results[ev]['cs'] = None
				go_charge_evs.remove(ev)

		station_id = open_stations[0]
		station_info = self.charging_stations.stations[station_id]
		resource_level = station_info["available_chargers"]

		random.shuffle(go_charge_evs)
		evs_to_assign = go_charge_evs[:min(resource_level, len(go_charge_evs))]
		total_assign = len(evs_to_assign)

		for ev in go_charge_evs:
			if ev in evs_to_assign:
	   
				added_SoC = max(1. - self.states["SoC"][ev], 0)
	
				if added_SoC >= self.evs.reserve_SoC*0.5:
					self.dispatch_results[ev]['cs'] = {"station_id": station_id} 
					self.dispatch_results[ev]['cs']["added_SoC"] = added_SoC
					self.dispatch_results[ev]['cs']["charging_duration"] = int(np.minimum(np.ceil(added_SoC/self.evs.charge_rate[ev]), self.delta1))

					charging_price = station_info["real_time_prices"][int(self.current_timepoint*self.dt//30)]
					self.dispatch_results[ev]['cs']["price"] = charging_price

					self.dispatch_results[ev]['cs']["session_cost"] = added_SoC * self.evs.b_cap[ev] * charging_price + station_info["one_time_fee"]
				
				else:
					self.dispatch_results[ev]['cs'] = None
					total_assign = np.maximum(total_assign-1,0)
			else:
				self.dispatch_results[ev]['cs'] = None

		self.charging_stations.adjust_occupancy(station_id, -total_assign)

	def update_gps_locations(self, ev_i):
		return (0, 0)


def main():
	example_config = {
		"total_time_steps": 216,
		"time_step_minutes":5,
		"total_evs": 5,
		"committed_charging_block_minutes": 15,
		"renewed_charging_block_minutes": 5, 
		"ev_params": None,
		"trip_params": None,
		"charging_params": None,
		"other_env_params": None
	}
	
	env = EVChargingEnv(example_config)  # 3 EVs, total 10 sessions, charging holds 2 sessions
	
	total_episodes = 10
	ep_pay = []
	ep_cost = []
	ep_returns = []
	ep_penalty = []
	for ep in range(total_episodes):
		env.reset()
		# env.show_config()

		for step in range(env.T):
			actions = env.action_space.sample()
			# actions = [0 for _ in range(env.N)]
			_, _, done, info, _ = env.step(actions)
			
			# if ep % 5 == 0:
			# 	env.report_progress()
	
			if done:
				break

		ep_pay.append(env.total_driver_pay_earned)
		ep_cost.append(env.total_charging_cost)
		ep_returns.append(env.ep_returns)
		ep_penalty.append(env.total_violation_penalty)
  
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
 
	ep_penalty = [round(float(r), 2) for r in ep_penalty]
	print("total penalty:", ep_penalty)
	print("average total penlaty is:", sum(ep_penalty)/total_episodes)	

	env.close()


if __name__ == "__main__":
	main()
