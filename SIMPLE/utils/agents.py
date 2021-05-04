import sys
from time import sleep

import numpy as np

np.set_printoptions(threshold=sys.maxsize)
import random
import string

from stable_baselines import logger


def sample_action(action_probs):
    action = np.random.choice(len(action_probs), p=action_probs)
    return action


def mask_actions(legal_actions, action_probs):
    masked_action_probs = np.multiply(legal_actions, action_probs)
    masked_action_probs = masked_action_probs / np.sum(masked_action_probs)
    return masked_action_probs


class Agent:
    def __init__(self, name, model=None, interact_with_probe=False):
        self.name = name
        self.id = self.name + '_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
        self.model = model
        self.points = 0

    def print_top_actions(self, action_probs):
        top5_action_idx = np.argsort(-action_probs)[:5]
        top5_actions = action_probs[top5_action_idx]
        logger.debug(
            f"Top 5 actions: {[str(i) + ': ' + str(round(a, 2))[:5] for i, a in zip(top5_action_idx, top5_actions)]}")

    def choose_action(self, env, choose_best_action, mask_invalid_actions):
        if self.name == 'human':
            if env.render_mode is "print":
                while True:
                    try:
                        action = int(input('\nPlease choose an action: '))
                        break
                    except ValueError:
                        pass
            else:
                co_ords = np.array(env.fig.ginput(n=1, timeout=99999999)).ravel()
                # all_sites = np.array(list(product([0, 1, 2], repeat=2)))
                all_sites = np.array([[0, 0],
                                      [1, 0],
                                      [2, 0],
                                      [0, 1],
                                      [1, 1],
                                      [2, 1],
                                      [0, 2],
                                      [1, 2],
                                      [2, 2]])
                all_sites_dist = np.sqrt(
                    np.square((co_ords[0] - all_sites[:, 0])) + np.square((co_ords[1] - all_sites[:, 1])))
                action = np.argmin(all_sites_dist)
                sleep(0.2)
            return int(action)
        elif self.name == 'rules':
            action_probs = np.array(env.rules_move())
            value = None
        else:
            action_probs = self.model.action_probability(env.observation)
            value = self.model.policy_pi.value(np.array([env.observation]))[0]
            logger.debug(f'Value {value:.2f}')

        self.print_top_actions(action_probs)

        # if mask_invalid_actions:
        # action_probs = mask_actions(env.legal_actions, action_probs)
        # logger.debug('Masked ->')
        #     # self.print_top_actions(action_probs)

        action = np.argmax(action_probs)
        logger.debug(f'Best action {action}')

        if not choose_best_action:
            action = sample_action(action_probs)
            logger.debug(f'Sampled action {action} chosen')

        return int(action)
