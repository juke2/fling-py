from typing import Optional


def create_board_from_point_list(points: list[tuple]) -> list[list[int]]:
    xmin, ymin = (
        min([point[0] for point in points]),
        min([point[1] for point in points]),
    )
    point_list = [[point[0] - xmin, point[1] - ymin] for point in points]
    xmax, ymax = (
        max([point[0] for point in point_list]),
        max([point[1] for point in point_list]),
    )
    board = [[0 for y in range(0, ymax + 1)] for x in range(0, xmax + 1)]
    for x, y in point_list:
        board[x][y] = 1
    return board


def pretty_tostr_board(
    board: list[list[int]],
    empty_icon: Optional[str] = None,
    fling_icon: Optional[str] = None,
) -> None:
    icon_0 = empty_icon if empty_icon else "0"
    icon_1 = fling_icon if fling_icon else "1"
    pretty_list = [""]
    for y in range(0, len(board[0])):
        for x in range(0, len(board)):
            pretty_list.append(icon_1 if board[x][y] == 1 else icon_0)
        pretty_list.append("\n")
    return " ".join(pretty_list)


board = create_board_from_point_list([(2, 3), (-1, 9)])


print(board)
print(pretty_tostr_board(board))
