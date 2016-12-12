import random
import json
from math import ceil

import gym

GAMES = ['air_raid', 'alien', 'amidar', 'assault', 'asterix',
         'asteroids', 'atlantis', 'bank_heist', 'battle_zone',
         'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout',
         'carnival', 'centipede', 'chopper_command', 'crazy_climber',
         'demon_attack', 'double_dunk', 'elevator_action', 'enduro',
         'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
         'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull',
         'kung_fu_master', 'montezuma_revenge', 'ms_pacman',
         'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
         'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank',
         'seaquest', 'skiing', 'solaris', 'space_invaders', 'star_gunner',
         'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture',
         'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']


GAME_NAMES = [''.join([g.capitalize() for g in game.split('_')])+'-v0'
              for game in GAMES]


class Agent(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, env, observation, reward):
        '''Called every time a new observation is available.
        Should return an action.
        '''
        return func(env, observation, reward)

    def episode(self):
        '''Called at the end of every episode'''

    @classmethod
    def wrap(cls, maybe_agent):
        if isinstance(cls, maybe_agent):
            return maybe_agent
        else:
            return cls(maybe_agent)


class RandomAgent(Agent):

    def __init__(self):
        pass

    def __call__(self, env, observation, reward):
        return env.action_space.sample()

    def episode(self):
        print 'Episode over'


class TestPlan(object):

    def __init__(self,
                 holdout_fraction=0.30,
                 regimen=1,
                 max_no_reward_turns=10000):
        '''Creates a TestPlan for running a set of games.

        `holdout_fraction` is a float for what percent of the overall games to
            keep out of training to be used as test games.
        `regimen` is the number of games in a row to play before switching
            to a new random game from the training set.
        `max_no_signal_turns` is how many turns to go in a row without any
            reward before artificially saying the game is "done"
        '''
        num_test_games = int(ceil(len(GAME_NAMES) * holdout_fraction))
        self.test_set = sorted(random.sample(GAME_NAMES, num_test_games))
        self.training_set = sorted(set(GAME_NAMES) - set(self.test_set))
        self.regimen = regimen
        self.max_no_reward_turns = max_no_reward_turns

    def save(self, filename):
        '''Save the TestPlan to a file'''
        filedata = {
            'test_set': self.test_set,
            'training_set': self.training_set,
            'regimen': self.regimen,
            'max_no_reward_turns': self.max_no_reward_turns,
        }
        with open(filename, 'w') as savefile:
            json.dump(savefile, filedata, sort_keys=True, indent=True)

    @staticmethod
    def load_from_file(filename):
        '''Load a TestPlan from a file'''
        with open(filename, 'r') as savefile:
            filedata = json.load(filename)
            plan = TestPlan()
            # Just overwrite the original fields. A little wasteful but w/e
            plan.test_set = filedata['test_set']
            plan.training_set = filedata['training_set']
            plan.regimen = filedata['regimen']
            plan.max_no_reward_turns = filedata['max_no_reward_turns']
        return plan

    def train(self, agent, render=False, max_episodes=-1):
        '''Training in a loop
        `agent` is the Agent to use to interact with the game
        `render` is whether to render game frames in real time
        `max_episodes` will limit episodes to run
        '''
        episodes = 0
        games_in_a_row = 0
        env = get_random_env(self.training_set)
        while max_episodes == -1 or episodes <= max_episodes:
            done = False
            reward = 0.0
            if games_in_a_row >= self.regimen:
                env = get_random_env(self.training_set)
                games_in_a_row = 0
            observation = env.reset()
            no_reward_for = 0

            while not done and no_reward_for <= self.max_no_reward_turns:
                if render:
                    env.render()
                action = agent(env, observation, reward)
                observation, reward, done, info = env.step(action)
                no_reward_for = no_reward_for + 1 if not reward else 0

            agent.episode()
            # update counts
            games_in_a_row += 1
            if episodes > -1:
                episodes += 1


def get_random_env(game_choices):
    game_name = random.choice(game_choices)
    print 'Going to play {} next'.format(game_name)
    env = gym.make(game_name)
    return env


def main():
    plan = TestPlan(holdout_fraction=0.30, regimen=1)

    agent = RandomAgent()

    plan.train(agent, render=False)


if __name__ == '__main__':
    main()
