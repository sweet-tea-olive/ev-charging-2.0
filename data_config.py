import os
import pandas as pd
import numpy as np
import json
from utils import df_to_list

def generate_config(
    total_time_steps=216,
    time_step_minutes=5,
    total_envs=5,  # Also used as the number of EVs (N)
    committed_charging_block_minutes=15,
    renewed_charging_block_minutes=5,
    charging_price_csv="price_30min_2020-06-01.csv",
    arrival_rates_fname="arrival_rates_5min_5evs_2019-04.csv",
    pay_rates_fname="pay_rates_5min_2019-04.csv",
    trip_records_fname="ready_trip_data2019-04.csv",
    init_SoCs_csv = "init_SoCs_10evs.csv",
    saved_trips_fname = "saved_trips2019-04.json"
):
    
    config = {
        "total_time_steps": total_time_steps,
        "time_step_minutes": time_step_minutes,
        "total_evs": total_envs,
        "committed_charging_block_minutes": committed_charging_block_minutes,
        "renewed_charging_block_minutes": renewed_charging_block_minutes,
        "ev_params": None,
        "trip_params": None,
        "charging_params": None,
        "other_env_params": None
    }

    config["charging_params"] = {
        1: {
            "location": (0, 0),
            "maximum_capacity": 5,
            "available_chargers": 5,
            "one_time_fee": 0,
            "occupy_per_min_cost": 0,
            "queue_per_min_cost": 0,
            "baseline_prices": df_to_list(charging_price_csv),
            "real_time_prices": df_to_list(charging_price_csv),
        }
    }

    config["ev_params"] = {
        "N": total_envs,
        "SoC_drop_per_min": [0.002] * total_envs,
        "SoC_added_per_min": [0.023] * total_envs,
        "battery_capacity_kwh": [50] * total_envs,
        "reserve_SoC": 0.1,
        "init_SoCs": df_to_list(init_SoCs_csv)
    }

    config["trip_params"] = {
        "arrival_rates_fname": arrival_rates_fname,
        "pay_rates_fname": pay_rates_fname,
        "saved_trips_fname": saved_trips_fname,
        "trip_records_fname": trip_records_fname
    }
    
    return config

def save_config(config, filename="config.json"):
    def convert(o):
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
    with open(filename, "w") as f:
        json.dump(config, f, default=convert, indent=4)
    print(f"Configuration successfully saved to {filename}")

if __name__ == "__main__":
    config = generate_config(
        total_time_steps= 216,
        time_step_minutes= 5,
        total_envs= 10,
        committed_charging_block_minutes=15,
        renewed_charging_block_minutes=5,
        charging_price_csv="price_30min_2020-06-01.csv",
        arrival_rates_fname="arrival_rates_5min_5evs_2019-04.csv",
        pay_rates_fname="pay_rates_5min_2019-04.csv",
        trip_records_fname="ready_trip_data2019-04.csv",
        init_SoCs_csv = "init_SoCs_10evs.csv",
        saved_trips_fname = "saved_trips2019-04.json"
    )
    save_config(config, "basic_config.json")
