import csv
import os
import sys
import time
from shutil import rmtree

import numpy as np
from mpi4py import MPI
from stable_baselines import logger
from stable_baselines.ppo1 import PPO1

import SIMPLE.config
from utils.register import get_network_arch


def write_results(players, game, games, episode_length):
    out = {'game': game
        , 'games': games
        , 'episode_length': episode_length
        , 'p1': players[0].name
        , 'p2': players[1].name
        , 'p1_points': players[0].points
        , 'p2_points': np.sum([x.points for x in players[1:]])
           }

    if not os.path.exists(SIMPLE.config.RESULTSPATH):
        with open(SIMPLE.config.RESULTSPATH, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=out.keys())
            writer.writeheader()

    with open(SIMPLE.config.RESULTSPATH, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out.keys())
        writer.writerow(out)

def load_model(env, name):
    filename = os.path.join(SIMPLE.config.MODELDIR, env.name, name)
    if os.path.exists(filename):
        logger.info(f'Loading {name}')
        ppo_model = PPO1.load(filename)
        # cont = True
        # while cont:
        #     try:
        #         ppo_model = PPO1.load(filename, env=env)
        #         return ppo_model
        #         cont = False
        #     except Exception as e:
        #         time.sleep(5)
        #         print(e)

    elif name == 'base.zip':
        cont = True
        while cont:
            try:

                rank = MPI.COMM_WORLD.Get_rank()
                if rank == 0:
                    ppo_model = PPO1(get_network_arch(env.name), env=env)
                    logger.info(f'Saving base.zip PPO model...')
                    ppo_model.save(os.path.join(SIMPLE.config.MODELDIR, env.name, 'base.zip'))
                else:

                    ppo_model = PPO1.load(os.path.join(SIMPLE.config.MODELDIR, env.name, 'base.zip'), env=env)

                cont = False
            except IOError as e:
                sys.exit(f'Permissions not granted on zoo/{env.name}/...')
            except Exception as e:
                print('Waiting for base.zip to be created...')
                time.sleep(2)

    else:
        raise Exception(f'\n{filename} not found')

    return ppo_model


def load_all_models(env, load=True):
    modellist = [f for f in os.listdir(os.path.join(SIMPLE.config.MODELDIR, env.name)) if f.startswith("_model")]
    modellist.sort()
    models = [load_model(env, 'base.zip')]
    if load:
        for model_name in modellist:
            models.append(load_model(env, name = model_name))
    else:
        for model_name in modellist:
            models.append(model_name)
    return models


def get_best_model_name(env_name):
    modellist = [f for f in os.listdir(os.path.join(SIMPLE.config.MODELDIR, env_name)) if f.startswith("_model")]

    if len(modellist) == 0:
        raise FileNotFoundError("Can't find any files")
        # filename = None
    else:
        modellist.sort()
        filename = modellist[-1]

    return filename


def get_model_stats(filename):
    if filename is not None:
        stats = filename.split('_')
        if len(stats) < 2:  # only model or best_model exists
            generation = int(stats[2])
            best_rules_based = float(stats[3])
            best_reward = float(stats[4])
            timesteps = int(stats[5])
            return generation, timesteps, best_rules_based, best_reward

    generation = 0
    timesteps = 0
    best_rules_based = -np.inf
    best_reward = -np.inf
    return generation, timesteps, best_rules_based, best_reward


def reset_files(model_dir):
    try:
        filelist = [f for f in os.listdir(SIMPLE.config.LOGDIR) if f not in ['.gitignore']]
        for f in filelist:
            if os.path.isfile(f):
                os.remove(os.path.join(SIMPLE.config.LOGDIR, f))

        for i in range(100):
            if os.path.exists(os.path.join(SIMPLE.config.LOGDIR, f'tb_{i}')):
                rmtree(os.path.join(SIMPLE.config.LOGDIR, f'tb_{i}'))

        open(os.path.join(SIMPLE.config.LOGDIR, 'log.txt'), 'a').close()

        filelist = [f for f in os.listdir(model_dir) if f not in ['.gitignore']]
        for f in filelist:
            os.remove(os.path.join(model_dir, f))
    except Exception as e:
        print(e)
        print('Reset files failed')
