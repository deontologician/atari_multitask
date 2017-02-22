import pytest

import amtlb


@pytest.fixture
def fake_games():
    return 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split()

class TestBenchmarkParms(object):

    @pytest.mark.parametrize('fold_num', [3, 5, 7, 11, 13])
    def test_folds_are_sane(self, fake_games, fold_num):
        bp = amtlb.BenchmarkParms(
            num_folds=fold_num,
            game_names=fake_games)
        fold_div = len(fake_games) // fold_num
        fold_rem = len(fake_games) % fold_num

        # Should create the number of folds requested
        assert len(bp.folds) == fold_num
        for fold in bp.folds:
            # No folds should be too large
            assert len(fold) in [fold_div, fold_div + 1]

        all_games_in_folds = set()
        for fold in bp.folds:
            all_games_in_folds.update(set(fold))
        assert set(fake_games) == all_games_in_folds
