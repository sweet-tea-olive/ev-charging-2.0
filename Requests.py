import random
import pandas as pd
import datetime
from data_loader import DataLoader

class Requests:
	def __init__(self, arrival_rates):
		self.requests_list = {}  # Dictionary to store requests by ID
		self.open_requests = []  # List to store IDs of open requests
		self.trip_records = None
		self.arrival_rates = arrival_rates
		self.pay_rates = None
		
	# Function to load trip records from CSV
	def load_trip_records(self, csv_file):
		self.trip_records = pd.read_csv(csv_file)

	def create_request(self, req_id, raised_time, pickup_location, trip_duration, driver_pay):
		# Creating a new request with necessary info
		self.requests_list[req_id] = {
			"raised_time": raised_time,
			"pickup_location": pickup_location,
			"trip_duration": trip_duration,
			"status": "open",  # Can be "open" or "completed"
			"driver_pay": driver_pay,
		}
		self.open_requests.append(req_id)  # Adding the new request ID to the open list

	def complete_request(self, req_id):
		# Mark the request as completed and remove from open_requests
		if req_id in self.requests_list:
			self.requests_list[req_id]["status"] = "completed"
			if req_id in self.open_requests:
				self.open_requests.remove(req_id)
	
	def update_open_requests(self, current_timepoint):
		# Filter open requests that are still within the valid time range (4 units in this case)
		self.open_requests = [
			req_id for req_id in self.open_requests
			if self.requests_list[req_id]["status"] == "open" and (current_timepoint - self.requests_list[req_id]["raised_time"] <=1)
		]
		return self.open_requests

	
	def sample_requests(self, num_requests, raised_time, max_trip_duration=10, pay_rate=1.0):

		num_requests = int(num_requests)

		if num_requests == 0:
			return None

		if self.pay_rates is not None:
			pay_rate = self.pay_rates[raised_time//5]
			

		highest_id = max(self.requests_list.keys(), default=-1)
		start_id = highest_id + 1

		sampled_requests = None
		if self.trip_records is not None:
			filtered_records = self.filter(raised_time)
			# print("filtered_records:", filtered_records.head())
			sampled_requests = filtered_records.sample(n=num_requests).to_dict('records')
			# print("sampled_requests:", sampled_requests)

		for i in range(num_requests):
			req_id = start_id + i
			# print("req_id:", req_id)
			if sampled_requests is not None:
				request_sample = sampled_requests[i]
				raised_time = raised_time
				pickup_location = (0,0)
				trip_duration = int(round(request_sample['trip_time']))
				# driver_pay = request_sample['driver_pay']
				driver_pay = pay_rate * trip_duration
				# print("trip_duration:", trip_duration)
			else:
				raised_time = raised_time
				pickup_location = (0, 0)
				trip_duration = random.randint(1, max_trip_duration)
				driver_pay = pay_rate * trip_duration

			self.create_request(req_id, raised_time, pickup_location, trip_duration, driver_pay)

		return [req_id for req_id in range(start_id, start_id + num_requests)]


	def filter(self, current_timepoint, start_hour=6):
	
		if self.trip_records is None:
			return pd.DataFrame()

		self.trip_records['request_datetime'] = pd.to_datetime(self.trip_records['request_datetime'])

		total_minutes = start_hour * 60 + current_timepoint
		hour = (total_minutes // 60)
		minute = (total_minutes % 60)
		
		# Filter the records to match the target time
		filtered_data = self.trip_records[self.trip_records['request_datetime'].dt.hour == hour]

		return filtered_data

def main():
	
	data_loader = DataLoader()
	
	arrival_rates = data_loader.load_arrival_rates("trip_data_5min_5evs2019-04.csv")
	requests = Requests(arrival_rates=arrival_rates)
	requests.load_trip_records("filtered_trip_data2019-04.csv")
	requests.pay_rates = data_loader.load_pay_rates("pay_rates_5min.csv")
	for current_timepoint in range(1080):
		if current_timepoint % 5 == 0:
			num_requests = requests.arrival_rates[current_timepoint//5]
			# print("num_requests:", num_requests)
			requests.sample_requests(num_requests, current_timepoint)

	print(len(requests.requests_list))
	# print(requests.requests_list)
 
	for key, value in requests.requests_list.items():
		hour = 6+(value['raised_time'] // 60)
		minute = (value['raised_time'] % 60)
		# print(f"Dictionary {key}: raised_time = {hour}:{minute}")

 

if __name__ == "__main__":
	main()