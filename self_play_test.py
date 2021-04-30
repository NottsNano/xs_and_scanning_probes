import os
from time import sleep

import numpy as np
import tensorflow as tf

from SIMPLE.utils.agents import Agent
from SIMPLE.utils.files import load_model
from SIMPLE.utils.register import get_environment
from SIMPLE.utils.selfplay import selfplay_wrapper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Settings
PLAYER_1_TYPE = "human" # 'best', 'mostly_best', 'random', 'human', 'rules', '<model>.zip'
PLAYER_2_TYPE = "best"
FIRST_PLAYER = "player_1" # 'player_1', 'player_2', 'random'
RENDER_MODE = "plot"  # None, 'print', 'plot'

# Setup environment
base_env = get_environment("tictactoe")
env = selfplay_wrapper(base_env)(opponent_type=PLAYER_2_TYPE, first_player=FIRST_PLAYER, render_mode=RENDER_MODE,
                                 verbose=False, deploy_mode="test")
env.seed(1234)

# Setup players
if "zip" in PLAYER_1_TYPE:
    player_0 = Agent('PPO_Agent', load_model(env, PLAYER_1_TYPE))
else:
    player_0 = Agent(PLAYER_1_TYPE)

done = False
obs = env.reset()

while not done:
    env.render()
    action = player_0.choose_action(env, choose_best_action=True, mask_invalid_actions=True)

    obs, reward, done, _ = env.unwrapped.step(action)

    if done:
        env.render()

# if env.check_game_over()[0] == 0 and env.check_game_over()[0] == True:
#     print("Draw!")
# else:
#     print(f"Winner = {env.current_player.token.symbol}")
