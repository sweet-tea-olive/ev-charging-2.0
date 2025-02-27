import numpy as np

"""
0.1 per session: fully depleted: 10 sessions
0.2 per session: fully charged: 5 session
	
"""


class EVs:
	def __init__(self, N, energy_consumption=0.002, battery_capacity=50, charge_rate=0.023, speed=0, reserve_SoC=0.1):
		self.N = N  # Number of EVs
		self.energy_consumption = N * [energy_consumption] # a list of N, per Delta_t reduced SoC
		self.charge_rate = N * [charge_rate] # a list of N, per session added SoC
		self.b_cap = N*[battery_capacity]  # a list of N, in kWh
		self.state = {
			"OperationalStatus": np.zeros(N, dtype=int),  # 0: idle, 1: serving, 2: charging
			"TimeToNextAvailability": np.zeros(N, dtype=int),
			# "SoC": np.random.uniform(0.2, 0.8, size=N),  # Initial SoC (randomized)
			"SoC": np.random.uniform(0.75, 0.8, size=N),  # Initial SoC (randomized)
			"Location": np.zeros((N, 2))  # Placeholder for GPS coordinates
		}
		self.speed = speed
		self.reserve_SoC = reserve_SoC

	def update_state(self, ev_index, new_state):
		
		self.state["OperationalStatus"][ev_index] = new_state[0]
		self.state["TimeToNextAvailability"][ev_index] = new_state[1]
		self.state["SoC"][ev_index] = new_state[2]
		self.state["Location"][ev_index] = new_state[3]

	def get_state(self, ev_index):
		state = [
      				self.state["OperationalStatus"][ev_index],
					self.state["TimeToNextAvailability"][ev_index],
					self.state["SoC"][ev_index],
					self.state["Location"][ev_index]
     			]
	
		return state
