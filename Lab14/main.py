# %%
import copy
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
from IPython.display import Image, display
from ple import PLE
from ple.games.flappybird import FlappyBird
from tqdm.autonotebook import tqdm, trange

os.environ["SDL_VIDEODRIVER"] = "dummy"  # this line disable pop-out window
game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)  # environment interface to game
env.reset_game()

# %%
# a dictionary describe state
"""
    player y position.
    players velocity.
    next pipe distance to player
    next pipe top y position
    next pipe bottom y position
    next next pipe distance to player
    next next pipe top y position
    next next pipe bottom y position
"""
game.getGameState()

# %%
MIN_EXPLORING_RATE = 0.01
MIN_LEARNING_RATE = 0.5


class Agent:
    def __init__(self, bucket_range_per_feature, num_action, t=0, discount_factor=0.99):
        self.update_parameters(t)  # init explore rate and learning rate
        self.q_table = defaultdict(lambda: np.zeros(num_action))
        self.discount_factor = discount_factor
        self.num_action = num_action

        # how to discretize each feature in a state
        # the higher each value, less time to train but with worser performance
        # e.g. if range = 2, feature with value 1 is equal to feature with value 0
        # bacause int(1/2) = int(0/2)
        self.bucket_range_per_feature = bucket_range_per_feature

    def select_action(self, state):
        # epsilon-greedy
        state_idx = self.get_state_idx(state)
        if np.random.rand() < self.exploring_rate:
            action = np.random.choice(self.num_action)  # Select a random action
        else:
            action = np.argmax(
                self.q_table[state_idx]
            )  # Select the action with the highest q
        return action

    def update_policy(self, state, action, reward, state_prime, action_prime):
        state_idx = self.get_state_idx(state)
        state_prime_idx = self.get_state_idx(state_prime)

        # Update Q_value using SARSA update rule
        self.q_table[state_idx][action] += self.learning_rate * (
            reward
            + self.discount_factor * self.q_table[state_prime_idx][action_prime]
            - self.q_table[state_idx][action]
        )

    def get_state_idx(self, state):
        # instead of using absolute position of pipe, use relative position
        state = copy.deepcopy(state)
        state["next_next_pipe_bottom_y"] -= state["player_y"]
        state["next_next_pipe_top_y"] -= state["player_y"]
        state["next_pipe_bottom_y"] -= state["player_y"]
        state["next_pipe_top_y"] -= state["player_y"]

        # sort to make list converted from dict ordered in alphabet order
        state_key = [k for k, v in sorted(state.items())]

        # do bucketing to decrease state space to speed up training
        state_idx = []
        for key in state_key:
            state_idx.append(int(state[key] / self.bucket_range_per_feature[key]))
        return tuple(state_idx)

    def update_parameters(self, episode):
        self.exploring_rate = max(
            MIN_EXPLORING_RATE, min(0.5, 0.99 ** ((episode) / 30))
        )
        self.learning_rate = max(MIN_LEARNING_RATE, min(0.5, 0.99 ** ((episode) / 30)))

    def shutdown_explore(self):
        # make action selection greedy
        self.exploring_rate = 0


# %%
num_action = len(env.getActionSet())
bucket_range_per_feature = {
    "next_next_pipe_bottom_y": 40,
    "next_next_pipe_dist_to_player": 512,
    "next_next_pipe_top_y": 40,
    "next_pipe_bottom_y": 20,
    "next_pipe_dist_to_player": 20,
    "next_pipe_top_y": 20,
    "player_vel": 4,
    "player_y": 16,
}
# init agent
agent = Agent(bucket_range_per_feature, num_action)


# %%
def make_anim(images, fps=60, true_image=False):
    duration = len(images) / fps

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip


# %%
reward_per_epoch = []
lifetime_per_epoch = []
exploring_rates = []
learning_rates = []
print_every_episode = 500
show_gif_every_episode = 5000
NUM_EPISODE = 40000

pbar = trange(NUM_EPISODE // print_every_episode, leave=True)
for episode in pbar:
    for _episode in trange(print_every_episode, leave=False):
        env.reset_game()

        state = game.getGameState()
        action = agent.select_action(state)

        frames = [env.getScreenRGB()]
        cum_reward = 0
        t = 0

        while not env.game_over():
            reward = env.act(env.getActionSet()[action])
            frames.append(env.getScreenRGB())
            cum_reward += reward

            state_prime = game.getGameState()
            action_prime = agent.select_action(state_prime)

            agent.update_policy(state, action, reward, state_prime, action_prime)

            state = state_prime
            action = action_prime
            t += 1

        agent.update_parameters(episode * print_every_episode + _episode)

    pbar.set_postfix(
        {
            "lifetime": t,
            "cum_reward": cum_reward,
            "exploring_rate": agent.exploring_rate,
            "learning_rate": agent.learning_rate,
        }
    )

    reward_per_epoch.append(cum_reward)
    exploring_rates.append(agent.exploring_rate)
    learning_rates.append(agent.learning_rate)
    lifetime_per_epoch.append(t)

    # for every 5000 episode, record an animation
    if episode % (show_gif_every_episode // print_every_episode) == 0:
        print("len frames:", len(frames))
        clip = make_anim(frames, fps=60, true_image=True).rotate(-90)
        display(clip.ipython_display(fps=60, autoplay=1, loop=1))
