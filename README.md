# Atari Multitask & Transfer Learning Benchmark (AMTLB)

This is a library to test how a reinforcement learning architecture
performs on all Atari games in OpenAI's gym. It performs two kinds of
tests, one for transfer learning and one for multitask
learning. Crucially, this benchmark tests how an *architecture*
performs. Training the architecture on games is part of the test, so
it does not test pre-trained networks (but see note below for
details).

Throughout this document we'll refer to the **architecture** as the
system being tested irrespective of individual weight values. We refer
to an **instance** being that architecture with a particular set of
weights (either trained or untrained). The benchmark trains several
instances of the architecture to judge the performance of the
architecture itself.

## Transfer learning benchmark
The goal of the transfer learning benchmark is to see how well the
architecture can learn a new game it has never seen before, just using
what it's learned from other games (so, how much knowledge is
transferred from one game to another).

The way it works is first we create a fresh instance of the
architecture (call it instance `A`), and then measure its score over
time as it sees ten million frames of a random Atari game (call it
game `T`). Next, we create another fresh instance of the architecture
(call it instance `B`), but this one we train on bunch of other Atari
games (but not on game `T` itself). Finally, we let `B` play ten
million frames of game `T` and measure its score over time.

For each time frame, we take the cumulative score of `A` and the
cumulative score of `B` and get the ratio `r = 1 - B / A`.

 * If `r` negative, then the architecture actually got worse from seeing other Atari games.
 * If `r` is about 0, then the architecture didn't really transfer knowledge well from having seen the other Atari games.
 * If `r` positive, then we're in the sweet spot and the architecture is successfully learning to play a new Atari game from other games.

We're not quite done though, because really this is just a measure of
how well the architecture did on game `T`. Some games may transfer
knowledge well, and other games may be so unlike other Atari games
that it's hard to transfer much knowledge. What we could do to get
around this is to then do the process above for each game in the
entire collection and average the scores.

This would take a really long time though, so as a compromise, instead
of just holding out one game in the above process, we hold out about
30% of all games as tests, and keep 70% of games for training. We then
do the above process to test, except we create a fresh instance for
each test game, and we save the state of network after it's been
trained on the training set of games. We reset it to that "freshly
trained" state before each test game (so it doesn't learn from the
other testing games). Then we shuffle the training and testing sets up
randomly and do this a few more times from scratch.

As an example, lets say there are five games `S`, `T`, `U`, `V`, `X`.

We'll measure the performance of a fresh instance on each of the games
for 10 million frames, getting `F_s`, `F_t`, `F_u`, `F_v`, and `F_x`
(`F` is for "fresh").

Then for the first trial, we'll randomly select `T` and `X` as the test games.
We'll train a new instance `F` on `S`, `T`, and `V` and save its weights as `F_suv`.
Then we train `F_suv` on `T` for ten million frames, getting `F_suv(T)`.
Then we train `F_suv` on `X` for ten million frames, getting `F_suv(X)`.

To get the score for the first trial, we average their ratios:

    r_1 = (F_suv(T)/F_t + F_suv(X)/F_x) / 2

Now we do a couple more trials, maybe using `S` and `T` as the test
games, then maybe for the third trial `X` and `S` as the tests (with a
small number of games like this, you probably don't want to pick
randomly, but for the number of Atari games available it's fine.

    r_2 = (F_uvx(S)/F_s + F_uvx(T)) / 2
    r_3 = (F_tuv(X)/F_x + F_tuv(S)) / 2

Finally, we average the scores from all three trials:

    r = (r_1 + r_2 + r_3) / 3

And `r` is the final transfer learning score for the architecture.

## Multitask learning benchmark

## Note on pre-training
