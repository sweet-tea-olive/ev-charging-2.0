import os
import pandas as pd
import numpy as np

class DataConfig:
    def __init__(self):
        self.data_loader = DataLoader()
        
        # Load all data here
        self.arrival_rates = self.data_loader.load_arrival_rates("trip_data_5min_5evs2019-04.csv")
        self.pay_rates = self.data_loader.load_pay_rates("pay_rates_5min.csv")
        self.ref_price_vector = self.data_loader.load_ref_price_vector("iso-2020-06-01-30min.csv")
        
    def get_arrival_rates(self):
        return self.arrival_rates

    def get_pay_rates(self):
        return self.pay_rates

    def get_ref_price_vector(self):
        return self.ref_price_vector