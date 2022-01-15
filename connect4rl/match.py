from typing import List, Tuple

from connect4rl import exceptions
from connect4rl.board import Board


def play_interactive_match():
    board = Board(rows=6, columns=7)
    board.display()
    while True:
        while True:
            col_id = int(input("Player one plays: "))
            try:
                has_won = board.player1_move(col_id)
                break
            except exceptions.BoardError as err:
                print(err)

        board.display()
        if has_won:
            print(f"We have a champion: Player one")
            break

        while True:
            col_id = int(input("Player two plays: "))
            try:
                has_won = board.player2_move(col_id)
                break
            except exceptions.BoardError as err:
                print(err)
        board.display()
        if has_won:
            print(f"We have a champion: Player two")
            break


def play_prepared_match(moves: List[Tuple[int, int]]):
    board = Board(rows=6, columns=7)
    board.display()

    for move_pair in moves:
        p1_move, p2_move = move_pair

        print(f"Player one plays: {p1_move}")
        has_won = board.player1_move(p1_move)
        board.display()
        if has_won:
            print(f"We have a champion: Player one")
            break

        print(f"Player one plays: {p2_move}")
        has_won = board.player2_move(p2_move)
        board.display()
        if has_won:
            print(f"We have a champion: Player two")
            break


if __name__ == "__main__":
    play_interactive_match()
