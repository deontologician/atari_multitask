import random
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


def setup():
    num_test_games = int(ceil(len(GAME_NAMES) * 0.30))
    test_games = sorted(random.sample(GAME_NAMES, num_test_games))
    training_games = sorted(set(GAME_NAMES) - set(test_games))
    return training_games, test_games


def random_play(game_names, strategy, per_game_limit=10000, render=False):
    while True:  # TODO: real end condition
        done = False
        reward = 0.0
        game_name = random.choice(game_names)
        print 'Going to play {} next'.format(game_name)
        env = gym.make(game_name)
        observation = env.reset()
        iterations = 0
        while not done and iterations <= per_game_limit:
            if render:
                env.render()
            action = strategy(env, observation, reward)
            observation, reward, done, info = env.step(action)
            if reward == 0.0:
                iterations += 1
            else:
                iterations = 0
        if iterations >= per_game_limit:
            print 'Ran too long with no reward'


def main():
    training_games, test_games = setup()
    strategy = lambda env, observation, reward: env.action_space.sample()

    random_play(training_games, strategy)


# Strategy
# Setup: select a subset of games as training games, rest are tests
# Train:
#  1. pick a random game from training games
#  2. use the agent to play the game until the game is done
#  3. Pick another random game, repeat.
# Test:
#  Strategy 1
#      - Train from training games
#      - Compare speed of learning on trained agent on unseen test games vs. untrained agent
#  Strategy 2.
#      - Train on all games
#      - Compare absolute score with world records


if __name__ == '__main__':
    main()
