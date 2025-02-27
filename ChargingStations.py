import random
import pandas as pd
import numpy as np
import os
from utils import df_to_list

class ChargingStations:
	def __init__(self, config=None):
		if config is None:
			config = {
				1: {
					"location": (0, 0),
					"maximum_capacity": 5,
					"available_chargers": 5,
					"one_time_fee": 0,
					"occupy_per_min_cost": 0,
					"queue_per_min_cost": 0,
					"baseline_prices": df_to_list("price_30min_2020-06-01.csv"),
					"real_time_prices": df_to_list("price_30min_2020-06-01.csv"),
				}
			}
		self.stations = config

	def reset(self, randomize_prices=True):
		
		self.open_stations = list(self.stations.keys())  # List of charger IDs with available slots >= 1
		
		if randomize_prices:# default to randomize prices for each brand new episode
			self.update_real_time_prices()

	def update_real_time_prices(self):
		for station_id in self.stations.keys():
			baseline_prices = np.array(self.stations[station_id]["baseline_prices"]) 

			# Calculate upper and lower bounds for price variation
			lb = np.max([np.min(baseline_prices) - np.abs(np.min(baseline_prices)) * 0.5, 0])  # Ensure lower bound is non-negative
			ub = np.max(baseline_prices) + np.abs(np.max(baseline_prices)) * 0.5
			
			# Add random noise to baseline prices
			noise = np.random.normal(0, 0.01, len(baseline_prices))

			# Clip the new prices between lb and ub, then store them
			self.stations[station_id]["real_time_prices"] = np.clip(baseline_prices + noise, lb, ub).tolist()


	def adjust_occupancy(self, station_id, increment):
		if station_id in self.stations:
			self.stations[station_id]["available_chargers"] += increment

	def update_open_stations(self):

		self.open_stations = [
			station_id for station_id, station_info in self.stations.items() if station_info["available_chargers"] >= 1
		]
		
		return self.open_stations
	
	def get_resource_level(self):
		
		total_available_chargers = sum(station_info['available_chargers'] for _, station_info in self.stations.items())
		
		return int(total_available_chargers)



def main():

	charging_system = ChargingStations()
	for current_timepoint in range(216):
		if current_timepoint % 6 == 0:
			charging_system.reset()
			price = charging_system.stations[1]["real_time_prices"][current_timepoint // 6]
			print(price)
   
if __name__ == "__main__":
	main()