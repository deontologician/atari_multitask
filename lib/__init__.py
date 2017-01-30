from __future__ import division

import random
import json
from math import ceil
from abc import ABCMeta, abstractmethod
import string
from datetime import datetime

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

NUM_GAMES = len(GAMES)


# The names above are easy to modify and read if we keep them
# separate, but to load the gym you need the name in camelcase with
# the version
def game_name(raw_name):
    return ''.join([g.capitalize() for g in game.split('_')]) + '-v0'

GAME_NAMES = [game_name(game) for game in GAMES]


def get_random_env(game_choices):
    game_name = random.choice(game_choices)
    print 'Going to play {} next'.format(game_name)
    env = gym.make(game_name)
    return env


def fold_name(num):
    name = string.ascii_uppercase[num % 26]
    if num > 25:
        name += str(num - 25)
    return name


class Agent(object):

    __metaclass__ = ABCMeta  # ensures subclasses implement everything

    @abstractmethod
    def __call__(self, observation, reward):
        '''Called every time a new observation is available.
        Should return an action in the action space.
        '''

    @abstractmethod
    def clone(self):
        '''Returns a copy of the agent and its weights.'''

    @classmethod
    @abstractmethod
    def load(cls, filename):
        '''Loads an agent (with weights) from a filename'''

    @abstractmethod
    def save(self, filename):
        '''Saves the agent (with weights) to a filename'''



class RandomAgent(Agent):
    '''Simple random agent that has no state'''

    def __call__(self, observation, reward):
        # The benchmark maps invalid actions to No-op (action 0)
        return random.randint(0, 17)

    def clone(self):
        return self  # RandomAgent has no state

    @classmethod
    def load(cls, filename):
        return cls()

    def save(self, filename):
        pass


class BenchmarkParms(object):
    def __init__(self,
                 num_folds=5,
                 frame_limit=10000000,
                 max_turns_w_no_reward=10000,
                 seed=None,
                 ):
        self.num_folds = num_folds
        self.frame_limit = frame_limit
        self.max_turns_w_no_reward = max_turns_w_no_reward
        self.seed = random.random() if seed is None else seed

        games = set(GAME_NAMES)
        fold_size = NUM_GAMES // num_folds
        remainder = NUM_GAMES % num_folds
        self.folds = [None] * num_folds

        for i in range(num_folds):
            if i < remainder:
                # distribute the remainder games evenly among the folds
                self.folds[i] = random.sample(games, fold_size + 1)
            else:
                self.folds[i] = random.sample(games, fold_size)
            games -= set(self.folds[i])

        assert(len(games) == 0)

    def save(self, filename):
        '''Save the TestPlan to a file'''
        filedata = {
            'folds': self.folds,
            'seed': self.seed,
            'frame_limit': self.frame_limit,
            'max_turns_w_no_reward': self.max_turns_w_no_reward,
        }
        with open(filename, 'w') as savefile:
            json.dump(filedata, savefile, sort_keys=True, indent=True)

    @staticmethod
    def load_from_file(filename):
        '''Load a BenchmarkParms from a file'''
        with open(filename, 'r') as savefile:
            filedata = json.load(savefile)
            parms = BenchmarkParms()
            # Just overwrite the original fields. A little wasteful but w/e
            parms.folds = filedata['folds']
            parms.num_folds = len(parms.folds)
            parms.frame_limit = filedata['frame_limit']
            parms.max_turns_w_no_reward = filedata['max_turns_w_no_reward']
            parms.seed = filedata['seed']
        return parms

class BenchmarkResult(object):
    def __init__(self, agent):
        self.agent = agent


class TransferBenchmark(object):
    '''Benchmark for testing knowledge transfer.

    Uses k-fold cross-validation to test an agent's performance on a
    new game. Each fold is a set of games held out from the training
    set.  All other folds except that one are the training
    set. Performance is compared as a ratio between the cumulative
    score over time of a fresh agent vs. the cumulative score over
    time of an agent trained on the training set. Both agents will be
    seeing the game in the test set for the first time, but one of the
    agents will have preparation, and the other will not. The ratio
    measures how much that preparation helps.

    For the purposes of this class, the term `game_agent` is used to
    denote a fresh agent who has no preparation, but has trained on a
    particular game. Game agents are identified by the name of their
    game. A `fold_agent` is an agent who has been trained up on all
    the games in the training set, i.e. everything but the test game
    fold.  The folds are indexed by an integer, so the fold_agents are
    also indexed by an integer indicating which fold is their test
    set (all other folds are implicitly their training set).

    In practice, we only care about the results of the game agents,
    and don't need to keep the trained agent around once we have their
    scores, since they are intended to be fresh. We do need to keep
    around the fold agents' trained agent since we should start from
    the same baseline on each test game. In other words, we shouldn't
    allow the agent to train on the other test games first, since then
    results would be dependent on the order in which we did tested
    games in the fold. So we checkpoint the agent at the time it
    finishes all of its training, and reset to that point before
    testing on each game in the test fold.

    '''
    def __init__(self, parms, AgentClass, dir=None):
        self.parms = parms  # BenchmarkParms
        self.AgentClass = AgentClass
        self.untrained_agent = AgentClass()
        self.game_agents = {}  # indexed by game name, since 1 per game
        self.game_results = {}  # benchmarks for each game agent
        self.fold_agents = []  # indexed by fold number
        self.fold_results = []  # Each fold gets a result for each game
        self.dir = dir or self.default_dir()
        # TODO: make dir recursively
        # TODO: add checkpoints, benchmark is ephemeral now

    def test_set(self, test_index):
        '''Copy the test set of game names from the BenchmarkParms'''
        return set(self.parms.folds[test_index])

    def training_set(self, test_index):
        '''Aggregate the training set from the current BenchmarParms'''
        return {x
                for i, fold in enumerate(self.parms.folds)
                for x in fold if i != test_index}

    def default_dir(self):
        '''A reasonable default directory to store benchmark results in'''
        time = datetime.now().strftime('%Y-%m-%d_%H.%M')
        return 'benchmarks/' + time + '/'

    def game_agent_filename(self, game_name):
        '''Constructs a save filename for a game agent'''
        return self.dir + 'game_agent_' + game_name

    def fold_agent_filename(self, fold_num):
        '''Constructs a save filename for a fold agent'''
        return self.dir + 'fold_agent_' + str(fold_num)

    def tested_agent_filename(self, fold_num, game_name):
        '''Constructs a filename for an fold agent tested on a particular
        game'''
        return self.dir + 'tested_agent_' + str(fold_num) + '_' + game_name

    def ensure_game_agents(self):
        '''Ensures there is a game agent for every game.

        Game agents don't cara about folds, and this function checks
        if an agent already exists, so there's no harm in running this
        function multiple times.
        '''
        for game_name in GAME_NAMES:
            if game_name not in self.game_agents:
                game_agent = self.untrained_agent.clone()
                self.game_results[game_name] = self.test(game_agent, game_name)
                game_agent.save(self.game_agent_filename(game_name))
                self.game_agents[game_name] = game_agent

    def do_folds(self):
        '''Runs through each fold, and trains an agent for each set.

        The outer loop of this function should run k times.
        '''
        self.ensure_game_agents()

        for fold_num in range(len(self.parms.folds)):
            training_set = self.training_set(fold_num)
            fold_agent = self.untrained_agent.clone()
            self.fold_agents[fold_num] = fold_agent
            self.train(fold_agent, training_set)
            fold_agent.save(self.fold_agent_filename(fold_num))

            test_set = self.test_set(fold_num)
            self.fold_results[fold_num] = fold_results = {}

            for game_name in test_set:
                tested_agent = fold_agent.clone()
                fold_results[game_name] = self.test(tested_agent, game_name)
                tested_agent.save(
                    self.tested_agent_filename(fold_num, game_name)



    # def train(self, agent, render=False, max_episodes=-1):
    #     '''Training in a loop
    #     `agent` is the Agent to use to interact with the game
    #     `render` is whether to render game frames in real time
    #     `max_episodes` will limit episodes to run
    #     '''
    #     episodes = 0
    #     games_in_a_row = 0
    #     env = get_random_env(self.training_set)
    #     while max_episodes == -1 or episodes <= max_episodes:
    #         done = False
    #         reward = 0.0
    #         if games_in_a_row >= self.regimen:
    #             env = get_random_env(self.training_set)
    #             games_in_a_row = 0
    #         observation = env.reset()
    #         no_reward_for = 0

    #         while not done and no_reward_for <= self.max_no_reward_turns:
    #             if render:
    #                 env.render()
    #             action = agent(env, observation, reward)
    #             observation, reward, done, info = env.step(action)
    #             no_reward_for = no_reward_for + 1 if not reward else 0

    #         agent.episode()
    #         # update counts
    #         games_in_a_row += 1
    #         if episodes > -1:
    #             episodes += 1


def main():
    bp = BenchmarkParms()

    architecture = RandomArchitecture()

    bench = Benchmark(bp, architecture)

    bench.train(render=False)


if __name__ == '__main__':
    main()


# TODO:
#  - [ ] Map from action_space to max_action space in agent action taking
#  - [ ] Rewrite test_plan to use the readme
#  - [ ] TransferBenchmark
#  - [ ] MultitaskBenchmark
#  - [ ] Ensure seeding works correctly
