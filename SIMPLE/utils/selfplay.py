import numpy as np
import random

import numpy as np
from stable_baselines import logger

from SIMPLE.utils.agents import Agent
from SIMPLE.utils.files import load_model, load_all_models, get_best_model_name


def selfplay_wrapper(env):
    class SelfPlayEnv(env):
        # wrapper over the normal single player env, but loads the best self play model
        def __init__(self, opponent_type, verbose, deploy_mode, first_player, render_mode=None):
            super(SelfPlayEnv, self).__init__(verbose)
            self.opponent_type = opponent_type
            self.opponent_models = load_all_models(self, load=(deploy_mode == "train"))
            self.best_model_name = get_best_model_name(self.name)
            self.render_mode = render_mode
            self.mode = deploy_mode
            self.first_player = first_player
            assert self.mode in ["train", "test"]

        def setup_opponents(self):
            if self.opponent_type == 'rules':
                self.opponent_agent = Agent('rules')
            else:
                # incremental load of new model
                best_model_name = get_best_model_name(self.name)
                if self.best_model_name != best_model_name:
                    self.opponent_models.append(load_model(self, best_model_name))
                    self.best_model_name = best_model_name

                if self.opponent_type == 'random':
                    start = 0
                    end = len(self.opponent_models) - 1
                    i = random.randint(start, end)
                    if type(self.opponent_models[i]) is str:
                        self.opponent_models[i] = load_model(env, name=self.opponent_models[i])
                    self.opponent_agent = Agent('ppo_opponent', self.opponent_models[i])

                elif self.opponent_type == 'best':
                    if type(self.opponent_models[-1]) is str:
                        self.opponent_models[-1] = load_model(env, name=self.opponent_models[-1])
                    self.opponent_agent = Agent('ppo_opponent', self.opponent_models[-1])

                elif self.opponent_type == 'mostly_best':
                    j = random.uniform(0, 1)
                    if j < 0.8:
                        if type(self.opponent_models[-1]) is str:
                            self.opponent_models[-1] = load_model(env, name=self.opponent_models[-1])
                        self.opponent_agent = Agent('ppo_opponent', self.opponent_models[-1])
                    else:
                        start = 0
                        end = len(self.opponent_models) - 1
                        i = random.randint(start, end)
                        if type(self.opponent_models[i]) is str:
                            # print(self.opponent_models[i], self.opponent_models[i], self.opponent_models[i])
                            self.opponent_models[i] = load_model(env, name=self.opponent_models[i])
                        self.opponent_agent = Agent('ppo_opponent', self.opponent_models[i])

                elif self.opponent_type == 'base':
                    self.opponent_agent = Agent('base', self.opponent_models[0])

            #
            if self.first_player == "random":
                self.agent_player_num = np.random.choice(self.n_players)
            elif self.first_player == "player_1":
                self.agent_player_num = 0
            elif self.first_player == "player_2":
                self.agent_player_num = 1
            else:
                raise ValueError("first_player not set correctly")

            self.agents = [self.opponent_agent] * self.n_players
            self.agents[self.agent_player_num] = None
            try:
                # if self.players is defined on the base environment
                logger.debug(f'Agent plays as Player {self.players[self.agent_player_num].id}')
            except:
                pass

        # def setup_opponents(self):
        #     if self.opponent_type == 'rules':
        #         self.opponent_agent = Agent('rules')
        #     else:
        #         # incremental load of new model
        #         best_model_name = get_best_model_name(self.name)
        #         if self.best_model_name != best_model_name:
        #             self.opponent_models.append(load_model(self, best_model_name ))
        #             self.best_model_name = best_model_name
        #
        #         if self.opponent_type == 'random':
        #             start = 0
        #             end = len(self.opponent_models) - 1
        #             i = random.randint(start, end)
        #             self.opponent_agent = Agent('ppo_opponent', self.opponent_models[i])
        #
        #         elif self.opponent_type == 'best':
        #             self.opponent_agent = Agent('ppo_opponent', self.opponent_models[-1])
        #
        #         elif self.opponent_type == 'mostly_best':
        #             j = random.uniform(0,1)
        #             if j < 0.8:
        #                 self.opponent_agent = Agent('ppo_opponent', self.opponent_models[-1])
        #             else:
        #                 start = 0
        #                 end = len(self.opponent_models) - 1
        #                 i = random.randint(start, end)
        #                 self.opponent_agent = Agent('ppo_opponent', self.opponent_models[i])
        #
        #         elif self.opponent_type == 'base':
        #             self.opponent_agent = Agent('base', self.opponent_models[0])
        #
        #     self.agent_player_num = np.random.choice(self.n_players)
        #     self.agents = [self.opponent_agent] * self.n_players
        #     self.agents[self.agent_player_num] = None
        #     try:
        #         #if self.players is defined on the base environment
        #         logger.debug(f'Agent plays as Player {self.players[self.agent_player_num].id}')
        #     except:
        #         pass

        def reset(self):
            super(SelfPlayEnv, self).reset()
            self.setup_opponents()

            if self.current_player_num != self.agent_player_num:
                self.continue_game()

            return self.observation

        @property
        def current_agent(self):
            return self.agents[self.current_player_num]

        def continue_game(self):
            observation = None
            reward = None
            done = None
            while self.current_player_num != self.agent_player_num:
                self.render()
                action = self.current_agent.choose_action(self, choose_best_action=False, mask_invalid_actions=False)
                observation, reward, done, _ = super(SelfPlayEnv, self).step(action)
                logger.debug(f'Rewards: {reward}')
                logger.debug(f'Done: {done}')
                if done:
                    break

            return observation, reward, done, {}

        def step(self, action):
            self.render(mode="print")
            observation, reward, done, _ = super(SelfPlayEnv, self).step(action)
            logger.debug(f'Action played by agent: {action}')
            logger.debug(f'Rewards: {reward}')
            logger.debug(f'Done: {done}')

            if not done:
                package = self.continue_game()
                if package[0] is not None:
                    observation, reward, done, _ = package

            agent_reward = reward[self.agent_player_num]
            logger.debug(f'\nReward To Agent: {agent_reward}')

            if done:
                self.render()

            return observation, agent_reward, done, {}

    return SelfPlayEnv
