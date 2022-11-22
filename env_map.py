import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class MapEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4) # 0 - up, 1 - right, 2 - down, 3 - left
        self.observation_space = spaces.Box(low=0,
                                            high=12,
                                            shape=(13,12),
                                            dtype=np.int16)
        self.reward_range = (-500, 500)
        self.current_episode = 0
        self.success_episode = []
        self.previous_choice = 0 # this is for the turret's movement
        self.total_steps = [] # this is to record the total steps per episode to use for analyses later
        self.player = [] # this is to record the total steps per episode to use for analyses later
        self.outcome = [] # this is to record the total steps per episode to use for analyses later


    def turret_direction(self, current_map, choice):
        up = np.array([[0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        right = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        down = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0]])

        left = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        turret_dict = {
                "up": up,
                "right": right,
                "down": down,
                "left": left
                }

        if self.current_step == 0: # still defining the board
            self.previous_choice = choice
            return current_map + list(turret_dict.values())[choice] # place turret in a random direction

        else: # turret can change direction
            # new map = current map - previous turret direction + new turret direction
            current_map = current_map - list(turret_dict.values())[self.previous_choice] + list(turret_dict.values())[choice] # change direction
            self.previous_choice = choice # set current choice to previous choice so the turret's direction can be changed correctly in the next step
            return current_map


    def place_hiding_agent(self):
        # 0 - open space
        # 1 - hiding agent -> will be randomly placed on a 0
        # 2 - hideout
        # 3 - turret & line of sight of turret - added before agent is placed
        # 4 - safe blocks
        map_plain = np.array([[0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 4, 2, 4, 0, 0, 0, 0, 2, 4, 4, 4],
                              [0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 2, 4],
                              [0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 2, 4],
                              [0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0],
                              [0, 4, 2, 2, 2, 0, 0, 0, 2, 2, 4, 0],
                              [0, 4, 4, 4, 2, 0, 0, 0, 2, 2, 4, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0]])

        choice = np.random.choice([0,1,2,3])
        world = self.turret_direction(map_plain, choice) # randomly place turret in facing either up or right or down or left
        
        # place hiding agent on a randomly selected 0 on the map
        zeros = np.argwhere(world == 0) # Indices where world == 0
        indices = np.ravel_multi_index([zeros[:, 0], zeros[:, 1]], world.shape) # Linear indices
        ind = np.random.choice(indices) # Randomly select a 0 to place the hiding agent
        world[np.unravel_index(ind, world.shape)] = 1 # place the hiding agent
        return world

    def reset(self): # starts a new episode - map is regenerated with random location for hiding agent and turret direction and reset all other params
        self.current_player = 1
        # P means the game is playable, W means somenone wins, L someone lose
        self.state = 'P'
        self.current_step = 0
        self.max_step = 300
        self.world = self.place_hiding_agent()
        return self._next_observation()

    def _next_observation(self):
        obs = self.world # agent can see everything
        obs = np.append(obs, [[self.current_player, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], axis=0)
        return obs

    def _take_action(self, action): # either agent or turret moves
        if self.current_player == 1: # agent
            current_pos = np.where(self.world == self.current_player)
            if action == 0: # up
                next_pos = (current_pos[0] - 1, current_pos[1])
                if next_pos[0] >= 0 and int(self.world[next_pos]) == 0: # agent can move here
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                elif next_pos[0] >= 0 and (int(self.world[next_pos]) == (1, 2)): # hideout or agent - agent cannot go here, loses chance to move
                    pass
                elif next_pos[0] >= 0 and (int(self.world[next_pos]) == 3): # turret & turret line of sight
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                    self.state = 'L' # agent loses
                elif next_pos[0] >= 0 and (int(self.world[next_pos]) == 4): # safe block
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                    self.state = 'W' # agent wins
            elif action == 1: # right
                next_pos = (current_pos[0], current_pos[1] + 1)
                if next_pos[1] < 3 and int(self.world[next_pos]) == 0: # agent can move here
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                elif next_pos[1] < 3 and int(self.world[next_pos]) in (1, 2): # hideout or agent - agent cannot go here, loses chance to move
                    pass
                elif next_pos[1] < 3 and (int(self.world[next_pos]) == 3): # turret & turret line of sight
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                    self.state = 'L' # agent loses
                elif next_pos[1] < 3 and (int(self.world[next_pos]) == 4): # safe block
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                    self.state = 'W' # agent wins
            elif action == 2: # down
                next_pos = (current_pos[0] + 1, current_pos[1])
                if next_pos[0] <= 3 and int(self.world[next_pos]) == 0: # agent can move here
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                elif next_pos[0] <= 3 and int(self.world[next_pos]) in (1, 2): # hideout or agent - agent cannot go here, loses chance to move
                    pass
                elif next_pos[0] <= 3 and (int(self.world[next_pos]) == 3): # turret & turret line of sight
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                    self.state = 'L' # agent loses
                elif next_pos[0] <= 3 and (int(self.world[next_pos]) == 4): # safe block
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                    self.state = 'W' # agent wins
            elif action == 3: # left
                next_pos = (current_pos[0], current_pos[1] - 1)
                if next_pos[1] >= 0 and int(self.world[next_pos]) == 0: # agent can move here
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                elif next_pos[1] >= 0 and int(self.world[next_pos]) in (1, 2): # hideout or agent - agent cannot go here, loses chance to move
                    pass
                elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 3): # turret & turret line of sight
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                    self.state = 'L' # agent loses
                elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 4): # safe block
                    self.world[next_pos] = self.current_player
                    self.world[current_pos] = 0
                    self.state = 'W' # agent wins
        else: # turret moves
            if action == 0: # up
                self.world = self.turret_direction(self.world, action) # change turret direction
                self.previous_choice = action
                if 1 not in self.world: # turret shot agent
                    self.state = 'W' # turret wins

            elif action == 1: # right
                self.world = self.turret_direction(self.world, action) # change turret direction
                self.previous_choice = action
                if 1 not in self.world: # turret shot agent
                    self.state = 'W' # turret wins

            elif action == 2: # down
                self.world = self.turret_direction(self.world, action) # change turret direction
                self.previous_choice = action
                if 1 not in self.world: # turret shot agent
                    self.state = 'W' # turret wins

            else: # left
                self.world = self.turret_direction(self.world, action) # change turret direction
                self.previous_choice = action
                if 1 not in self.world: # turret shot agent
                    self.state = 'W' # turret wins


    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        # uncomment below to generate figures to produce gifs of episodes - WARNING this significantly increases the runtime.
        #print(self.world)
        #plt.matshow(self.world, cmap="inferno")
        #time = str(datetime.now().strftime('%H-%M-%S-%f'))
        #plt.savefig(f'images/map_{time}_step_{str(self.current_step)}.png')
        #plt.close()
        if self.state == 'W':
            print(f'Player {self.current_player} won after {self.current_step} steps')
            if self.current_player == 3: # if turret shoots agent, small negative reward - this encourages the turret to not avoid shooting the agent
                reward = -1
            else:
                reward = 500 # if the agent wins - big reward
            done = True
        elif self.state == 'L': # agent loses by stepping into the turret's line of fire
            print(f'Player {self.current_player} lost after {self.current_step} steps')
            reward = -500 # big penalty for losing
            done = True
        elif self.state == 'P': # still playable
            reward = -2 # encourages agent to hide as quickly as possible, and the turret to shoot if possible
            done = False
            
        if self.current_step >= self.max_step: # end episode if more than the maximum amount of steps are taken
            done = True

        if done:
            self.total_steps.append(self.current_step) # save data
            self.player.append(self.current_player) # save data
            self.outcome.append(self.state) # save data
            #self.render_episode(self.state) # nice output but not necessary
            self.current_episode += 1

        if self.current_player == 1: # change players
            self.current_player = 3
        else:
            self.current_player = 1

        obs = self._next_observation()
        return obs, reward, done, {}

    def get_player(self):
        return self.player

    def get_total_steps(self):
        return self.total_steps

    def get_outcome(self):
        return self.outcome

    #def render_episode(self, win_or_lose):
    #    self.success_episode.append(
    #    'Success' if win_or_lose == 'W' else 'Failure')
    #    file = open('render/render.txt', 'a')
    #    file.write(' — — — — — — — — — — — — — — — — — — — — — -\n')
    #    file.write(f'Episode number {self.current_episode}\n')
    #    file.write(f'{self.success_episode[-1]} in {self.current_step} steps\n')
    #    file.close()