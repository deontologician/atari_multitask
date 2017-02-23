from functools import partial
from random import randint, sample
from string import uppercase, lowercase
import json
from cStringIO import StringIO


import pytest
from mock import patch, MagicMock

import amtlb


@pytest.fixture
def game_names():
    return list(sample(uppercase, randint(14, 26)))

@pytest.fixture(params=[1, 3, 5, 7, 11, 13])
def num_folds(request):
    return request.param

@pytest.fixture
def benchmark_parms(game_names, num_folds):
    return partial(
        amtlb.BenchmarkParms,
        game_names=game_names,
        num_folds=num_folds,
    )

@pytest.fixture
def random_benchmark_parms(game_names):
    return partial(amtlb.BenchmarkParms,
        num_folds=randint(0, 11),
        max_rounds_w_no_reward=randint(0, 1000),
        seed=randint(0, 10000),
        game_names=game_names,
    )

@pytest.fixture
def random_filename():
    return ''.join(sample(lowercase, 10))

@pytest.fixture
def random_file(random_filename):
    fileobj = StringIO()
    fake_open = MagicMock(return_value=fileobj)
    with patch('__builtin__.open', new=fake_open):
        yield fileobj

def folds_equal(A, B):
    _A = {frozenset(a) for a in A}
    _B = {frozenset(b) for b in B}
    return _A == _B

class TestBenchmarkParms(object):

    def test_creates_right_num_folds(self, num_folds, benchmark_parms):
        bp = benchmark_parms()
        assert len(bp.folds) == num_folds

    def test_folds_are_all_close_in_size(
            self, game_names, num_folds, benchmark_parms):
        bp = benchmark_parms()

        fold_div = len(game_names) // num_folds
        fold_rem = len(game_names) % num_folds

        for fold in bp.folds:
            assert len(fold) in [fold_div, fold_div + 1]

    def test_all_games_go_in_a_fold(self, game_names, benchmark_parms):
        bp = benchmark_parms()

        all_games_in_folds = set()
        for fold in bp.folds:
            all_games_in_folds.update(set(fold))
        assert set(game_names) == all_games_in_folds

    def test_save_roundtrip(
            self, random_benchmark_parms, random_file, random_filename):
        bp = random_benchmark_parms()
        bp.save(random_filename)

        file_contents = random_file.getvalue()

        j = json.loads(file_contents)

        assert j.pop('num_folds') == bp.num_folds
        assert j.pop('max_rounds_w_no_reward') == bp.max_rounds_w_no_reward
        assert j.pop('seed') == bp.seed
        assert j.pop('max_rounds_per_game') == bp.max_rounds_per_game
        assert j.pop('game_names') == bp.game_names
        assert folds_equal(j.pop('folds'), bp.folds)
        assert not j
