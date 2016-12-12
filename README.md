# Atari multitask

This is a library to run a reinforcement learning agent through all Atari games available through OpenAI's gym. The idea is to be a gauntlet designed for testing multi-task learning in a fairly unforgiving (perhaps brutal) manner.

The high level differences in this library are:

    1. Cares about how your agent trains
    2. Wants you to be learning during the test phase

To achieve these goals, the library:

    1. Gives your agent a new random game after every episode
    2. Gives the agent no out-of-band signal that the game is over and the next has started (other than a negative final reward)
    3. Treats the task as "how fast can you learn a new atari game".
    4. Holds out a "test set" of games the agent won't see during training.
    5. The test is to see how well the agent learns the new games with a limited number of attempts.
