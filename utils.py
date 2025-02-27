import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def visualize_trajectory(agents_trajectory):

    N, T = agents_trajectory["Reward"].shape

    time_steps = range(T + 1)

    plt.figure(figsize=(15, 10))

    for n in range(N):
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, agents_trajectory['SoC'][n, :], marker='o', linestyle='-', label=f'Agent {n}')
        plt.title('Battery State of Charge Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Battery SoC')
        plt.legend(loc='upper right')

        plt.subplot(3, 1, 2)
        plt.plot(time_steps, agents_trajectory['TimeToNextAvailability'][n, :], marker='o', linestyle='-', label=f'Agent {n}')
        plt.title('Time to Next Availability Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Time to Availability')
        plt.legend(loc='upper right')

        plt.subplot(3, 1, 3)
        plt.plot(time_steps, agents_trajectory['OperationalStatus'][n, :], marker='o', linestyle='-', label=f'Agent {n}')
        plt.title('Operational Status Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Status (0=Idle, 1=Serving, 2=Charging)')
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def print_ev_rewards_summary(Reward):
    
		total_rewards = Reward.sum(axis=1)
		for ev, reward in enumerate(total_rewards):
			print(f"EV {ev}: Total Reward = {reward}")

		max_ev = total_rewards.argmax()
		min_ev = total_rewards.argmin()
		print(f"EV {max_ev} earned the most with {total_rewards[max_ev]}.")
		print(f"EV {min_ev} earned the least with {total_rewards[min_ev]}.")

def plot_scores(scores, log_scale=False):
    plt.figure(figsize=(10, 5))
    if log_scale:
        log_scores = np.log(scores)
        plt.plot(log_scores, label='Log Scores')
        plt.yscale('log')
        plt.title('Scores and Log Scores Over Episodes')
    else:
        plt.plot(scores, label='Scores')
        plt.title('Scores Over Episodes')
    
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()
    


# 6 6:15
# 0 1

# 0/(60/15) = 0 ... 0
# 1/(60/15) = 0 ... 1
# 2/(60/15) = 0 ... 2
# 3/(60/15) = 0 ... 3
# 4/(60/15) = 1 ... 0
# to_price_idx: current_timepoint/(60 // 15)



