from enum import Enum, auto
from pprint import pprint
from typing import List, Tuple

from connect4rl import exceptions

Position = Tuple[int, int]


class State(Enum):
    empty = auto()
    player1 = auto()
    player2 = auto()


class Board:
    def __init__(self, columns: int = 6, rows: int = 7):
        self.rows = rows
        self.columns = columns
        self._grid = self._init_grid(self.columns)
        self._win_len = 4

    @staticmethod
    def _init_grid(width) -> List[List[State]]:
        return [[] for _ in range(width)]

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
        column_list = self._grid[column]
        if len(column_list) > self.rows:
            raise exceptions.BoardError(f"Column {column} is already full")
        column_list.append(player)
        return column, len(column_list) - 1

    def check_victory(self, position: Position, player: State) -> bool:
        horizontal = (
            1
            + self._count_coin(position, player, (1, 0))
            + self._count_coin(position, player, (-1, 0))
        )

        vertical = 1 + self._count_coin(position, player, (0, -1))
        diag_right = (
            1
            + self._count_coin(position, player, (1, 1))
            + self._count_coin(position, player, (-1, -1))
        )
        diag_left = (
            1
            + self._count_coin(position, player, (-1, 1))
            + self._count_coin(position, player, (1, -1))
        )

        if (
            horizontal >= self._win_len
            or vertical >= self._win_len
            or diag_right >= self._win_len
            or diag_left >= self._win_len
        ):
            return True
        return False

    def _count_coin(
        self, position: Position, player: State, traslation: Position
    ) -> int:
        col, row = position
        trasl_col, trasl_row = traslation
        col += trasl_col
        row += trasl_row

        if col < 0 or row < 0:
            return 0

        try:
            player_coin = self._grid[col][row]
        except IndexError:
            return 0

        if player_coin == player:
            return 1 + self._count_coin((col, row), player, traslation)
        return 0

    def display(self):
        pprint(self._grid)
