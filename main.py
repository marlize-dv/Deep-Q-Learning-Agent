from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN
from env_map import MapEnv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # to stop library version clash errors - might not be needed

models_dir = "models/DQN"
logdir = "logs"
imdir = "images"
renderdir = "render"

# create subfolders if they do not exist
if not os.path.exists(models_dir): 
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(imdir):
    os.makedirs(imdir)
if not os.path.exists(renderdir):
    os.makedirs(renderdir)

env = MapEnv() # create an instance of the map environment

# define the deep q-learning model, write logs to tensorboard, set agent to explore for the first 60% of the total episodes, start learning after
# 5000 "burn-in" episodes, set learning rate for DQN to 0.0001, and gamma (exploration vs exploitation factor) to 0.4 to encourage the agent to find
# the best reward sooner rather than later.
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir, exploration_fraction=0.6, learning_starts=5000, learning_rate=0.0001, gamma=0.4)

TIMESTEPS = 4000000
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
model.save(f"{models_dir}/{TIMESTEPS}")

env.close()
total_steps = env.get_total_steps()
player = env.get_player()
outcome = env.get_outcome()

# save some of the data to do analyses
results_df = pd.DataFrame({'player':player, 'outcome':outcome, 'total_steps':total_steps})
results_df.to_csv("results.csv") # save for later if needed

print(len(results_df[(results_df["player"] == 1) & (results_df["outcome"] == "W")]) / len(results_df)) # % of time that player 1 wins