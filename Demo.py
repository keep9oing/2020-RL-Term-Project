###############
#   DEMO V1   #
###############

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import numpy as np
import copy
import random

# Environment
class Env():
    def __init__(self):
        self.uav = [0,0]
        self.recharger = [2,2]
        self.goal = [4,4]

        self.map_size = 5
        self.world = np.zeros((self.map_size, self.map_size))

        self.total_fuel = 5
        self.current_fuel = self.total_fuel

        self.reset()

    def reset(self):
        self.uav=[0,0]
        self.current_fuel = self.total_fuel
        self.fuel_level = [[1] if i<self.current_fuel else [0] for i in range(self.total_fuel)]
        self.world = np.zeros((self.map_size, self.map_size))
        
        self.world[self.goal[0]][self.goal[1]] = 3
        self.world[self.recharger[0]][self.recharger[1]] = 2
        self.world[self.uav[0]][self.uav[1]] = 1

        initial_state = [self.uav[0], self.uav[1], self.current_fuel]

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
            next_state = [self.uav[0], self.uav[1], self.current_fuel]
            reward = -1
            done = False


            # FUEL Value change
            if self.uav == self.recharger:
                self.world[prev_state[0]][prev_state[1]] = 0
                self.current_fuel = min(self.total_fuel, self.current_fuel+1)
            else:
                if action == 4:
                    self.world[prev_state[0]][prev_state[1]] = 1
                else:
                    self.world[prev_state[0]][prev_state[1]] = 0
                self.current_fuel -= 1
            
            # FUEL level change for ploting
            self.fuel_level = [[1] if i<self.current_fuel else [0] for i in range(self.total_fuel)]

            # Just for rendering
            if prev_state == self.recharger and self.uav != self.recharger:
                self.world[self.recharger[0]][self.recharger[1]] = 2
            elif prev_state == self.recharger and action == 4:
                self.world[self.recharger[0]][self.recharger[1]] = 1

            # OUT OF FUEL
            if self.current_fuel <= 0:
                print("out of fuel")
                reward = -100
                done = True
            
            # GOAL reached
            if self.uav == self.goal:
                print("GOAL!!!")
                reward = 100
                done = True
            
        else:
            print('out of the world')
            next_state = None
            reward = -100
            done = True
        
        return next_state, reward, done

env = Env()

# World Template
cmap1 = colors.ListedColormap(['White', 'Black', 'Green', 'Yellow'])
cmap2 = colors.ListedColormap(['White', 'Blue'])

fig = plt.figure(figsize=(8,6))
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

episode = 100
max_step = 100

# action: 0-> right, 1-> left, 3-> up, 4-> down, 5-> stay
action_dict = {0:"right", 1:"left", 2:"up", 3:"down", 4:"stay"}

action_list = [0,0,3,3,4,4,4,4,4,3,3,0,0,0]

for epi in range(episode):
    print("EPI START")
    _ = env.reset()
    render(env)

    step = 0
    rewards = []
    done = False    
    
    while (not done) and (step < max_step):
        action = action_list[step]
        print(action_dict[action])

        next_s, reward, done = env.get_action(action)
        step += 1
        rewards.append(reward)

        render(env)

    print("episode:",epi,"avg reward:",sum(rewards)/len(rewards))
plt.show()

#TODO: GIT,  Q-learning