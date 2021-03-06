###############
#  REINFORCE  #
###############

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import numpy as np
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Environment
class Env():
    def __init__(self):
        self.uav = [0,0]
        self.rechargers = [[2,5],[6,3],[7,7]]
        self.goal = [9,9]

        self.map_size = 10
        self.world = np.zeros((self.map_size, self.map_size))

        self.total_fuel = 7
        self.current_fuel = self.total_fuel
        self.current_recharger = None

        self.reset()

    def reset(self):
        self.uav=[0,0]
        self.current_fuel = self.total_fuel
        self.fuel_level = [[1] if i<self.current_fuel else [0] for i in range(self.total_fuel)]
        self.world = np.zeros((self.map_size, self.map_size))
        
        self.world[self.goal[0]][self.goal[1]] = 3
        for i in range(len(self.rechargers)):
            self.world[self.rechargers[i][0]][self.rechargers[i][1]] = 2
        self.world[self.uav[0]][self.uav[1]] = 1

        initial_state = (self.uav[0], self.uav[1], self.current_fuel)

        return initial_state

    def get_action(self, action):

        prev_state = copy.deepcopy(self.uav)

        # action: 0-> right, 1-> left, 3-> up, 4-> down, 5-> stay
        if action == 0:
            self.uav[1] += 1
        elif action == 1:
            self.uav[1] -= 1
        elif action == 2:
            self.uav[0] -= 1
        elif action == 3:
            self.uav[0] += 1
        else:
            pass
        
        if self.uav[0] in range(self.world.shape[0]) and self.uav[1] in range(self.world.shape[1]):

            self.world[self.uav[0]][self.uav[1]] = 1
            next_state = (self.uav[0], self.uav[1], self.current_fuel)
            reward = -1
            done = False

            # FUEL Value change
            if self.uav in self.rechargers:
                self.world[prev_state[0]][prev_state[1]] = 0
                self.current_fuel = min(self.total_fuel, self.current_fuel+1)
                self.current_recharger = self.rechargers[self.rechargers.index(self.uav)]
            else:
                if action == 4:
                    self.world[prev_state[0]][prev_state[1]] = 1
                else:
                    self.world[prev_state[0]][prev_state[1]] = 0
                self.current_fuel -= 1
            
            # FUEL level change for ploting
            self.fuel_level = [[1] if i<self.current_fuel else [0] for i in range(self.total_fuel)]

            # Overlapping
            if prev_state in self.rechargers and self.uav not in self.rechargers:
                for i in range(len(self.rechargers)):
                    self.world[self.rechargers[i][0]][self.rechargers[i][1]] = 2
            elif prev_state in self.rechargers and action==4:
                self.world[self.current_recharger[0]][self.current_recharger[1]] = 1

            # OUT OF FUEL
            if self.current_fuel <= 0:
                print("out of fuel")
                reward = -10
                done = True
            
            # GOAL reached
            if self.uav == self.goal:
                print("GOAL!!!")
                reward = 10
                done = True
            
        else:
            print('out of the world')
            next_state = None
            reward = -1000
            done = True
        
        d2g = np.linalg.norm((np.array(self.uav) - np.array(self.goal)))
        #d2g_prev = np.linalg.norm((np.array(prev_state) - np.array(self.goal)))

        if self.current_fuel <= np.sum(np.abs((np.array(self.uav) - np.array(self.goal)))) and self.uav in self.rechargers:
            reward += 50

        #if d2g < d2g_prev:
        reward -= d2g*10
        return next_state, reward, done

class REINFORCE(nn.Module):
    def __init__(self, device, state_space, action_space, learning_step, discount_factor, epsilon):
        super(REINFORCE, self).__init__()

        self.device = device

        self.data = []
        
        self.s_space = len(state_space)
        self.a_space = action_space[0]
        self.alpha = learning_step
        self.gamma = discount_factor
        self.epsilon = epsilon

        self.fc1 = nn.Linear(self.s_space, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, self.a_space)
        self.optimizer = optim.SGD(self.parameters(), lr=self.alpha)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x

    def put_data(self, item):

        self.data.append(item)
    
    def update(self):

        R = 0
        
        self.optimizer.zero_grad()
        
        for r, prob in self.data[::-1]:
            R = r + self.gamma * R
            loss = -torch.log(prob) * R
            loss.to(self.device).backward()
        # print(R)
        self.optimizer.step()
        self.data = []

env = Env()

# World Template
cmap1 = colors.ListedColormap(['White', 'Black', 'Green', 'Yellow'])
cmap2 = colors.ListedColormap(['White', 'Blue'])

fig = plt.figure(1,figsize=(8,6))
gs = GridSpec(1, 2, width_ratios=[4, 1])
ax1 = fig.add_subplot(gs[0])
ax1.set_title('World')
ax1.axis('off')

ax2 = fig.add_subplot(gs[1])
ax2.set_yticks(np.arange(0,env.total_fuel+1,step=1))
ax2.set_xticks([])
ax2.pcolormesh(env.fuel_level, cmap=cmap2, edgecolors='k', linewidths=3, vmin=0, vmax=1)
ax2.set_title('Fuel level')


def render(env_):
    ax1.pcolormesh(env_.world[::-1], cmap=cmap1, edgecolors='k', linewidths=3, vmin=0, vmax=3)
    ax2.pcolormesh(env.fuel_level, cmap=cmap2, edgecolors='k', linewidths=3, vmin=0, vmax=1)
    plt.pause(1)

episode = 50000
max_step = 100
state_space = [5,5,6] 
action_space = [5]
learning_step = 0.0001
discount_factor = 0.99
epsilon = 0.5

render_ep = 4000
render_flag = True

# action: 0-> right, 1-> left, 3-> up, 4-> down, 5-> stay
action_dict = {0:"right", 1:"left", 2:"up", 3:"down", 4:"stay"}

if torch.cuda.is_available():
        device = torch.device("cuda")

agent = REINFORCE(device, state_space, action_space, learning_step, discount_factor, epsilon).to(device)

reward_log = []

for epi in range(episode):
    # print("EPI START")
    state = env.reset()

    if epi % render_ep == 0 and render_flag:
        render(env)

    step = 0
    total_reward = 0
    done = False    
    
    while (not done) and (step < max_step):

        action_prob = agent(torch.tensor(state).float().to(device))
        # print(action_prob)

        m = Categorical(action_prob)
        action = m.sample()

        next_state, reward, done = env.get_action(action)

        agent.put_data((reward, action_prob[action]))

        state = next_state
        step += 1
        total_reward += reward

        if epi % render_ep == 0 and render_flag:
            render(env)

    agent.update()

    print("episode:",epi,"avg reward:", total_reward)
    reward_log.append(total_reward)

rf = plt.figure(2)
plt.plot(reward_log)

plt.close(fig=fig)
plt.show()

# import pickle

# with open('asset/reward_log/LARGE/REINFORCE.pkl', 'wb') as f:
#     pickle.dump(reward_log, f)
