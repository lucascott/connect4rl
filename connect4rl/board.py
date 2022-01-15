from enum import IntEnum
from pprint import pprint
from typing import Tuple

import numpy as np

from connect4rl import exceptions

Position = Tuple[int, int]


class State(IntEnum):
    empty = 0
    player1 = 1
    player2 = 2


class Board:
    def __init__(self, rows: int = 6, columns: int = 7):
        self._grid = np.full((rows, columns), fill_value=State.empty)
        self._win_seq_length = 4

    @property
    def rows(self) -> int:
        return self._grid.shape[0]

    @property
    def columns(self) -> int:
        return self._grid.shape[1]

    def player1_move(self, column):
        player = State.player1
        position = self._play_move(player, column)
        return self.check_victory(position, player)

    def player2_move(self, column):
        player = State.player2
        position = self._play_move(player, column)
        return self.check_victory(position, player)

    def _play_move(self, player: State, column) -> Position:
        if not (0 <= column < self.columns):
            raise exceptions.BoardError(f"Column {column} is not valid")
        empty_slots: np.ndarray = np.nonzero(self._grid[:, column] == 0)[0]
        if not empty_slots.size:
            raise exceptions.BoardError(f"Column {column} is already full")
        first_empty_idx = empty_slots[0]
        self._grid[first_empty_idx, column] = player
        return first_empty_idx, column

    def check_victory(self, position: Position, player: State) -> bool:
        horizontal = (
            1
            + self._count_coin(position, player, (0, 1))
            + self._count_coin(position, player, (0, -1))
        )

        vertical = 1 + self._count_coin(position, player, (-1, 0))
        diag_right = (
            1
            + self._count_coin(position, player, (1, 1))
            + self._count_coin(position, player, (-1, -1))
        )
        diag_left = (
            1
            + self._count_coin(position, player, (1, -1))
            + self._count_coin(position, player, (-1, 1))
        )

        if (
            horizontal >= self._win_seq_length
            or vertical >= self._win_seq_length
            or diag_right >= self._win_seq_length
            or diag_left >= self._win_seq_length
        ):
            return True
        return False

    def _count_coin(
        self, position: Position, player: State, traslation: Position
    ) -> int:
        row, col = position
        trasl_row, trasl_col = traslation
        row += trasl_row
        col += trasl_col

        if col < 0 or row < 0:
            return 0

        try:
            player_coin = self._grid[row, col]
        except IndexError:
            return 0

        if player_coin == player:
            return 1 + self._count_coin((row, col), player, traslation)
        return 0

    def display(self):
        pprint(self._grid[::-1, :])
