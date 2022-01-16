import random

import numpy as np

from connect4rl.board import Board


class Connect4Rl:
    def __init__(self, rows, columns):
        self.board = Board(rows, columns)
        self._current_player = 1
        self.reward = 1

    def reset(self):
        self._current_player = 1
        self.board = Board(self.board.rows, self.board.columns)

    def _get_open_columns(self):
        empty_slots = np.any(self.board.status() == 0, axis=0)
        return empty_slots.nonzero()[0]

    def step(self, action):

        has_won = self.board.player1_move(action)
        if has_won:
            return None, self.reward, True, {}
        open_cols = self._get_open_columns()
        has_won = self.board.player2_move(random.choice(open_cols))
        if has_won:
            return None, -self.reward, True, {}

        return self.board.status(), 0, False, {}
