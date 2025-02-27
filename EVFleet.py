import numpy as np

class EVFleet:
	def __init__(self, config=None):
		if config is None:
			config = {
				"N": 5,
				"SoC_drop_per_min": [0.002] * 5,
				"SoC_added_per_min": [0.023] * 5,
				"battery_capacity_kwh": [50] * 5,
				"reserve_SoC": 0.1,
				"init_SoCs": [0.8] * 5
			}

		self.N = config.get("N", 5)
		self.energy_consumption = config.get("SoC_drop_per_min", [0.002] * self.N)
		self.charge_rate = config.get("SoC_added_per_min", [0.023] * self.N)
		self.b_cap = config.get("battery_capacity_kwh", [50] * self.N)
		self.reserve_SoC = config.get("reserve_SoC", 0.1)
		self.init_SoCs = np.array(config.get("init_SoCs", [0.8] * self.N))

		self.all_states = None  # Initialize placeholder

	def reset(self, randomize_init_SoCs=True):
		if randomize_init_SoCs:
			self.init_SoCs = np.random.uniform(0.75, 0.8, size=self.N)

		self.all_states = {
			"OperationalStatus": np.zeros(self.N, dtype=int),  # 0: idle, 1: serving, 2: charging
			"TimeToNextAvailability": np.zeros(self.N, dtype=int),
			"SoC": self.init_SoCs.copy(),  # Ensure a fresh copy
			"Location": np.zeros((self.N, 2))  # Placeholder for GPS coordinates
		}
		return self.all_states

	def update_state(self, ev_index, new_state):
	
		self.all_states["OperationalStatus"][ev_index] = new_state[0]
		self.all_states["TimeToNextAvailability"][ev_index] = new_state[1]
		self.all_states["SoC"][ev_index] = new_state[2]
		self.all_states["Location"][ev_index] = new_state[3]

	def get_state(self, ev_index):
    
		return [
				self.all_states[key][ev_index] 
				for key in ["OperationalStatus", "TimeToNextAvailability", "SoC", "Location"]
			]

	def get_all_states(self):
		
		return self.all_states
