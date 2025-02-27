import pandas as pd

class DataLoader:
	@staticmethod
	def load_trip_records(file_path):
		return pd.read_csv(file_path)

	@staticmethod
	def load_price_records(file_path):
		return pd.read_csv(file_path)

	@staticmethod
	def load_ref_price_vector(file_path):
		return pd.read_csv(file_path).values.flatten()

	@staticmethod
	def load_arrival_rates(file_path):
		return pd.read_csv(file_path).values.flatten()
	
	@staticmethod
	def load_pay_rates(file_path):
		return pd.read_csv(file_path).values.flatten()