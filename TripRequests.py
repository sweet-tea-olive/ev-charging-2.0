import random
import pandas as pd
import datetime
from utils import df_to_list

class TripRequests:
	def __init__(self, config=None):
		if config is None:
			config = {
				"arrival_rates_fname": "arrival_rates_5min_5evs_2019-04.csv",
				"pay_rates_fname": "pay_rates_5min_2019-04.csv",
				"saved_trips_fname": None,
				"trip_records_fname": "ready_trip_data2019-04.csv"
			}
	 
		self.arrival_rates = df_to_list(config.get("arrival_rates_fname"))
		self.pay_rates = df_to_list(config.get("pay_rates_fname"))
		self.saved_data = df_to_list(config.get("saved_trips_fname"))
		self.trip_records = pd.read_csv(config.get("trip_records_fname"))

	def reset(self, randomize_trips=True):
		self.requests = {}
		self.open_requests = []
		self.max_id = 0

		if randomize_trips is False:
			if self.saved_data is not None:
				pickup_location = (0, 0)
				n_rows = len(self.saved_data)
				n_cols = len(self.saved_data[0])
				for i in range(n_rows):
					raised_time = i
					for j in range(n_cols):
						trip_duration = self.saved_data[i][j]
						driver_pay = trip_duration * self.pay_rates[j]
						self.create_request(raised_time, pickup_location, trip_duration, driver_pay)
			else:
				raise Exception("Unable to use saved trip data: No saved data available.")

					
	def create_request(self, raised_time, pickup_location, trip_duration, driver_pay):
		self.max_id += 1
		self.requests[self.max_id] = {
			"raised_time": raised_time,
			"pickup_location": pickup_location,
			"trip_duration": trip_duration,
			"status": "open",  # Can be "open" or "completed"
			"driver_pay": driver_pay,
		}
		self.open_requests.append(self.max_id)

	def complete_request(self, req_id):
		if req_id in self.requests and req_id in self.open_requests:
			self.requests[req_id]["status"] = "completed"
			self.open_requests.remove(req_id)
		else:
			raise Exception("Request does not exist.")
	
	def update_open_requests(self, current_timepoint):
		# Filter open requests that are still within the valid time range (4 units in this case)
		self.open_requests = [
			req_id for req_id in self.open_requests
			if self.requests[req_id]["status"] == "open" and (current_timepoint - self.requests[req_id]["raised_time"] <= 1)
		]
		return self.open_requests

	def sample_requests(self, num_requests, raised_time, max_trip_duration=10, pay_rate=1.0):
		num_requests = int(num_requests)

		if num_requests == 0:
			return None

		if self.pay_rates is not None:
			pay_rate = self.pay_rates[raised_time]

		sampled_requests = None
		if self.trip_records is not None:
			filtered_records = self.filter(raised_time)
			sampled_requests = filtered_records.sample(n=num_requests).to_dict('records')

			if sampled_requests is not None:
				for v in sampled_requests:
					raised_time = raised_time
					pickup_location = (0, 0)
					trip_duration = int(round(v['trip_time']))
					driver_pay = pay_rate * trip_duration
					self.create_request(raised_time, pickup_location, trip_duration, driver_pay)
		else:
			for i in range(num_requests):
				raised_time = raised_time
				pickup_location = (0, 0)
				trip_duration = random.randint(1, max_trip_duration)
				driver_pay = pay_rate * trip_duration
				self.create_request(raised_time, pickup_location, trip_duration, driver_pay)

	def filter(self, current_timepoint, start_hour=6):
		if self.trip_records is None:
			return pd.DataFrame()

		self.trip_records['request_datetime'] = pd.to_datetime(self.trip_records['request_datetime'])

		total_minutes = start_hour * 60 + current_timepoint
		hour = (total_minutes // 60)
		minute = (total_minutes % 60)
		
		filtered_data = self.trip_records[self.trip_records['request_datetime'].dt.hour == hour]
		return filtered_data


def main():
	

	requests = TripRequests()
	requests.reset()
	for current_timepoint in range(216):
		num_requests = requests.arrival_rates[current_timepoint]
		requests.sample_requests(num_requests, current_timepoint)

	print(len(requests.requests))
	print(sum(requests.arrival_rates))

	print("Time when trips are requested:")
	for key, value in requests.requests.items():
		hour = (6*60+value['raised_time']*5) // 60
		minute = (6*60+value['raised_time']*5) % 60
		print(f"{hour}:{minute}")

 

if __name__ == "__main__":
	main()