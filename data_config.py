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
    baseline_prices=None,
    real_time_prices=None,
    arrival_rates_fname="arrival_rates_5min_5evs_2019-04.csv",
    pay_rates_fname="pay_rates_5min_2019-04.csv",
    trip_records_fname="ready_trip_data2019-04.csv",
    init_SoCs=None,
    saved_trips_fname="saved_trips2019-04.json"
):
    """
    Generate a configuration dictionary for the EV charging environment.

    Parameters:
        total_time_steps (int): Total simulation steps.
        time_step_minutes (int): Minutes per simulation step.
        total_envs (int): Number of EVs (also set as 'N' in ev_params).
        committed_charging_block_minutes (int): Duration of a committed charging block.
        renewed_charging_block_minutes (int): Duration of a renewed charging block.
        baseline_prices (list, optional): List of baseline prices; if None, loads default CSV.
        real_time_prices (list, optional): List of real-time prices; if None, loads default CSV.
        arrival_rates_fname (str): Filename for the arrival rates CSV.
        pay_rates_fname (str): Filename for the pay rates CSV.
        trip_records_fname (str): Filename for the trip records CSV.
        init_SoCs (list, optional): List of initial state-of-charge values; if None, loads default CSV.
        saved_trips_fname (str or None): Filename for saved trip data, or None.

    Returns:
        dict: The configuration dictionary.
    """
    # Set defaults for list parameters if they are not provided.
    if baseline_prices is None:
        baseline_prices = df_to_list("price_30min_2020-06-01.csv")
    if real_time_prices is None:
        real_time_prices = df_to_list("price_30min_2020-06-01.csv")
    if init_SoCs is None:
        init_SoCs = df_to_list("init_SoCs_10evs.csv")

    config = {
        "total_time_steps": total_time_steps,
        "time_step_minutes": time_step_minutes,
        "total_evs": total_envs,
        "committed_charging_block_minutes": committed_charging_block_minutes,
        "renewed_charging_block_minutes": renewed_charging_block_minutes,
        "ev_params": None,
        "trip_params": None,
        "charging_params": None,
        "other_env_params": None,
    }

    # Define charging parameters (using station 1 as the example)
    config["charging_params"] = {
        1: {
            "location": (0, 0),
            "maximum_capacity": 5,
            "available_chargers": 5,
            "one_time_fee": 0,
            "occupy_per_min_cost": 0,
            "queue_per_min_cost": 0,
            "baseline_prices": baseline_prices,
            "real_time_prices": real_time_prices,
        }
    }

    # Define EV parameters
    config["ev_params"] = {
        "N": total_envs,
        "SoC_drop_per_min": [0.002] * total_envs,
        "SoC_added_per_min": [0.023] * total_envs,
        "battery_capacity_kwh": [50] * total_envs,
        "reserve_SoC": 0.1,
        "init_SoCs": init_SoCs,
    }

    # Define trip parameters
    config["trip_params"] = {
        "arrival_rates_fname": arrival_rates_fname,
        "pay_rates_fname": pay_rates_fname,
        "saved_trips_fname": saved_trips_fname,
        "trip_records_fname": trip_records_fname,
    }
    
    return config

def save_config(config, filename="config.json"):
    """
    Save the configuration dictionary to a JSON file.

    Parameters:
        config (dict): The configuration dictionary.
        filename (str): The filename to save the configuration.
    """
    def convert(o):
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
    with open(filename, "w") as f:
        json.dump(config, f, default=convert, indent=4)
    print(f"Configuration successfully saved to {filename}")

if __name__ == "__main__":
    # # Generate a configuration for training.
    # # For training, we might leave 'real_time_prices' and 'init_SoCs' empty (or None)
    # # so that the environment can generate random data each episode.
    # train_config = generate_config(
    #     total_time_steps=216,
    #     time_step_minutes=5,
    #     total_envs=10,
    #     committed_charging_block_minutes=15,
    #     renewed_charging_block_minutes=5,
    #     baseline_prices=df_to_list("price_30min_2020-06-01.csv"),
    #     real_time_prices=[],  # Use empty list for training: random generation will occur.
    #     arrival_rates_fname="arrival_rates_5min_5evs_2019-04.csv",
    #     pay_rates_fname="pay_rates_5min_2019-04.csv",
    #     trip_records_fname="ready_trip_data2019-04.csv",
    #     init_SoCs=[],  # Use empty list for training.
    #     saved_trips_fname=None,
    # )
    # save_config(train_config, "config_for_train.json")
    
    # Generate a configuration for evaluation.
    # For evaluation, we need to provide fixed data for reproducibility:
    # - real_time_prices: a list of 36 price points (for 18 hours with 30-min resolution)
    # - init_SoCs: a list for all EVs.
    # - saved_trips_fname: the filename for saved trip data is used.
    # Additionally, arrival rates, pay rates, and trip records are not needed for evaluation.
    eval_config = generate_config(
        total_time_steps=216,
        time_step_minutes=5,
        total_envs=10,
        committed_charging_block_minutes=15,
        renewed_charging_block_minutes=5,
        baseline_prices=[],  # No baseline prices needed for evaluation.
        real_time_prices=df_to_list("price_30min_2020-06-01.csv"),
        arrival_rates_fname=None,
        pay_rates_fname=None,
        trip_records_fname=None,
        init_SoCs=df_to_list("init_SoCs_10evs.csv"),
        saved_trips_fname="saved_trips2019-04.json",
    )
    save_config(eval_config, "config_for_eval_1.json")
