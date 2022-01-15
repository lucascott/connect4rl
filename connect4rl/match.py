from connect4rl import exceptions
from connect4rl.board import Board


def play_match():
    board = Board(columns=6, rows=7)
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


if __name__ == "__main__":
    play_match()
