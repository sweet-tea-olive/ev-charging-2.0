import random
import pandas as pd
import numpy as np
import os
from data_loader import DataLoader

class ChargingStations:
	def __init__(self, ref_price_vector=[0.01] * 36):
		self.stations_list = {}
		self.open_stations = []
		self.occupy_fee = 100
		self.queue_fee = 100
		self.ref_price_vector = ref_price_vector
		self.low_price = self.ref_price_vector - np.abs(np.min(self.ref_price_vector))*0.2
		self.high_price = self.ref_price_vector + np.abs(np.max(self.ref_price_vector))*0.2

	def load_price_records(self, station_id, csv_file):
		self.stations_list[station_id]['price_records'] = pd.read_csv(csv_file)
		
	def sample_prices(self):

		noise = np.random.normal(0, 0.01, len(self.ref_price_vector))
		sampled_prices = self.ref_price_vector + noise
		sampled_prices = np.clip(sampled_prices, self.low_price, self.high_price)
		return sampled_prices
	
	def random_register(self, station_id):

		if station_id not in self.stations_list:

			self.stations_list[station_id] = {
				"maximum capacity": 2,
				"location": (0, 0),
				"number_of_open": 2, 
				"one_time_fee": 0,
				"forecasted_prices": self.ref_price_vector,
				"price": self.ref_price_vector[0]
			}

		else:
			raise ValueError(f"Station ID {station_id} already exists.")
		

	def update_all_prices(self):
		"""Sample prices for all registered stations based on their own reference price vectors."""
		for station_id in self.stations_list.keys():
			self.stations_list[station_id]["forecasted_prices"] = self.sample_prices()

		
	def adjust_occupancy(self, station_id, increment):
		if station_id in self.stations_list:
			self.stations_list[station_id]["number_of_open"] += increment

	def update_open_stations(self):
		# List of charger IDs with available slots >= 1
		self.open_stations = [
			station_id for station_id, station_info in self.stations_list.items() if station_info["number_of_open"] >= 1
		]
		
		return self.open_stations
	
	def get_resource_level(self):
		
		total_open = sum(station_info['number_of_open'] for _, station_info in self.stations_list.items())
		
		return int(total_open)



def main():
	data_loader = DataLoader()

	ref_price_vector = data_loader.load_ref_price_vector("iso-2020-06-01-30min.csv")
	charging_system = ChargingStations(ref_price_vector)
	charging_system.random_register(station_id=1)
	for current_timepoint in range(216):
		if current_timepoint % 6 == 0:
			charging_system.update_all_prices()
			price = charging_system.stations_list[1]["forecasted_prices"][current_timepoint // 6]
			print(price)
   
if __name__ == "__main__":
	main()