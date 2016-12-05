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


GAME_NAMES = [''.join([g.capitalize() for g in game.split('_')])
              for game in GAMES]


def main():
    import gym
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    env.render()


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
#  Stretegy 2.
#      - Train on all games
#      - Compare absolute score with world records


if __name__ == '__main__':
    main()
