from matplotlib import pyplot as plt
import pickle
import numpy as np

# with open('asset/reward_log/Large/Q-learning.pkl', 'rb') as f:
#     Q_rewards = pickle.load(f)

# with open('asset/reward_log/Simple/SARSA.pkl', 'rb') as f:
#     SARSA_rewards = pickle.load(f)

with open('asset/reward_log/Revised/REINFORCE.pkl', 'rb') as f:
    REINFORCE_rewards = pickle.load(f)

duration = 100
average = [sum(REINFORCE_rewards[i:i+duration])/duration for i in range(len(REINFORCE_rewards)-duration)]
plt.plot(REINFORCE_rewards)
# plt.plot(SARSA_rewards)
# plt.plot(REINFORCE_rewards)
plt.plot(average)



plt.title("Total reward per episode")
plt.xlabel("episode")
plt.ylabel("total reward")
plt.grid()
plt.legend(['REINFORCE','average'])

plt.show()