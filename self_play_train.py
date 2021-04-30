import os

import tensorflow as tf
from stable_baselines import PPO1
from stable_baselines.common.callbacks import EvalCallback

import SIMPLE.config
from SIMPLE.utils.callbacks import SelfPlayCallback, BeholderCallback
from SIMPLE.utils.register import get_environment
from SIMPLE.utils.selfplay import selfplay_wrapper
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

base_env = get_environment("tictactoe")
env = selfplay_wrapper(base_env)(opponent_type="mostly_best", verbose=True, deploy_mode="train", first_player="random")
env.seed(1234)

params = {'gamma': 0.99
    , 'timesteps_per_actorbatch': 1024
    , 'clip_param': 0.2
    , 'entcoeff': 0.1
    , 'optim_epochs': 4
    , 'optim_stepsize': 0.0003
    , 'optim_batchsize': 1024
    , 'lam': 0.95
    , 'adam_epsilon': 1e-05
    , 'schedule': 'linear'
    , 'verbose': 1
    , 'tensorboard_log': SIMPLE.config.LOGDIR
          }

model = PPO1.load('SIMPLE/zoo/pretrained/tictactoe/base.zip', env, **params)
# model = PPO1(env, **params)

callback_args = {
    'eval_env': selfplay_wrapper(base_env)(opponent_type="mostly_best", verbose=True, deploy_mode="test", first_player="random",),
    'best_model_save_path': SIMPLE.config.TMPMODELDIR,
    'log_path': SIMPLE.config.LOGDIR,
    'eval_freq': 10240,
    'n_eval_episodes': 100,
    'deterministic': False,
    'render': False,
    'verbose': 0,
    "callback_on_new_best": EvalCallback(
        eval_env=selfplay_wrapper(base_env)(opponent_type='rules', verbose=True, deploy_mode="test", first_player="random"),
        eval_freq=1,
        n_eval_episodes=100,
        deterministic=False,
        render=False,
        verbose=0
    )}

# sess = K.get_session()
# beholder_callback = BeholderCallback(K.get_session(), SIMPLE.config.LOGDIR, sess=sess)

eval_callback = SelfPlayCallback("mostly_best", 0.2, "tictactoe", **callback_args)
model.learn(total_timesteps=int(1e9), callback=[eval_callback], reset_num_timesteps=False, tb_log_name="tb")

env.close()
