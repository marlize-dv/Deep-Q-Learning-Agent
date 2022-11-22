# Deep Q-Learning Agent

Deep Q-learning is a type of reinforcement learning based on normal Q-learning, but in stead of using a Q-table that contains the maximum expected reward for each possible action in the current state, a neural network is used to determine the expected reward for each possible action in the current state.

For this task, there is a map with four "hideouts", one hiding agent, and a turret from which the agent should learn to hide.

![supplied map](https://user-images.githubusercontent.com/91065960/203115003-1d7c163b-bba9-476f-a983-c5a915aafdda.png)

In my model, there are two 'agents' that can take one of four possible actions in any state. 
The first agent is the agent whose goal is to hide (called player 1 in my code). 
As long as the agent stays in the map, the actions that this agent can take is to:
- move one step up
- move one step right
- move one step down
- move one step left

If the agent attempts to make an illegal move (such as trying to move out of the map) it loses its turn and the turret moves.
The second agent is the turret (called player 3 in my code), whose goal is to shoot the hiding agent.
The turret can also take any one of the four possible actions:
- shoot up
- shoot right
- shoot down
- shoot left

To simplify the task, I defined "safe spaces" behind the hideouts where the hiding agent is safe.
Below is a description of the map:

<img src="https://user-images.githubusercontent.com/91065960/203115434-50bd696c-5ca9-4ba6-8964-2ca612b50e98.png" width="500">

An episode is over when:
- the hiding agent finds a safe space (hiding agent wins).
- the hiding agent moves into the turret's current line of fire (hiding agent loses).
- the turret turns its line of fire onto the hiding agent (turret wins).
- the number of steps in an episode reach 300.

## This project code

The project consists of two files: `main.py` used to run the code and train the agents and `env_map.py` which contains all of the reinforcement learning things, such as the environment map, the actions, and the rewards.
The code used in `env_map.py` is inspired by and adapted from [this](https://medium.com/analytics-vidhya/how-to-create-a-custom-gym-environment-with-multiple-agents-f368d13582ee) article written by [Mathieu Cesbron](https://medium.com/@mathieuces).
The code in `main.py` is a combination of the same article and the tutorials in [this](https://www.youtube.com/playlist?list=PLQVvvaa0QuDf0O2DWwLZBfJeYY-JOeZB1) YouTube playlist.
The requirements to be able to reproduce everything is contained in `requirements.txt`.
To reproduce these results, the folder where you store the code should have the following sub-folders:

- logs (if you want to visualize the training with TensorBoard)
- models (if you want to save the model)
- images (if you want to produce gifs of the agent navigating through the map)
- render (if you want to follow the training with outputs in a text file)

and finally the two code files in the same folder:

- main.py
- env_map.py


## Some results and conclusion

Below are some visualizations of how the agent navigates through the map, that were taken early on in the training, so the hiding agent's behavior is not optimal yet, and the exploration rate was still very high.
On the left is an example of the agent winning the episode, and on the right is an example of the agent losing the episode.
If the agent stands still it is because it is trying to make an illegal move, for example, out of the map or onto a hideout block.

<img src="https://user-images.githubusercontent.com/91065960/203115879-396a0ce5-5725-40cc-93c6-e508fe5ec227.gif" width="400"> <img src="https://user-images.githubusercontent.com/91065960/203115892-e8f3250e-32ac-42a5-b34b-1fbdb39a562c.gif" width="400">

Since the hiding agent is placed randomly on an open space in the map at the start of each episode and there are four "safe spaces", I am assuming that it is a bit hard to "learn" the environment.
This is because perhaps the agent is placed next to a "safe spaces" and finds it quickly in one episode, and in the next episode, it is placed close to a different "safe space", but out of experience, the agent will want to return to the "safe space" that it found in the previous episode.
The goal of setting the `exploration_fraction = 0.6` is to try and overcome this, but it did not seem to work the way I had hoped it would.
I also set the `gamma = 0.4` to incentivize the agent to find the reward faster. 
The mean total steps taken in an episode was about 18 steps (considering that each movement by the hiding agent is counted as a step and each movement by the turret is counted as a separate step).
By looking at the TensorBoard logs, it is visible how the number of steps per episode decreased as the number of episodes increased:

<img src="https://user-images.githubusercontent.com/91065960/203293764-471a5aa3-57f3-42cb-9afd-6eb540237a15.png" width="420">

Finally, after 4 million episodes, the percentage of episodes that were won by the hiding agent was only 54.5%. 
I am assuming this is due to the hiding agent and turret working with the same reward function - i.e. if the hiding agent wins, the total reward increases, and if the turret wins, the total reward reduces.
To solve this I think that I should have created a separate class for the hiding agent and a separate class for the turret, each with their own reward functions, but due to time constraints this was not possible.
I know that the agent has learnt something since the average number of steps per episode decreased over the 4 million episodes.
From the TensorBoard logs I also obtained the following:

<img src="https://user-images.githubusercontent.com/91065960/203118579-45015c1e-df9a-41d8-b650-01a1b0f55730.png" width="420"> <img src="https://user-images.githubusercontent.com/91065960/203118565-57e13272-313f-41fd-8352-2ec218a07cef.png" width="400">

On the left is the mean reward per episode, and it is clear that it initially increased and eventually plateaud as the number of episodes increased.
Similarly, the loss of the DQN algorithm initially decreased and also eventually plateaud as the number of episodes increased.
