"""
Module containing tools to solve the game \"Fling!\" \n
\"Big Fling doesn't want you to know that flings can be flung\"
"""

from typing import Optional, Union
import time
import copy
import random
import altair as alt
import pandas as pd


def create_board_from_point_list(points: list[tuple[int, int]]) -> list[list[int]]:
    points = process_points_list(points)
    xmin, ymin = (
        min([point[0] for point in points]),
        min([point[1] for point in points]),
    )
    point_list = [[point[0] - xmin, point[1] - ymin, point[2]] for point in points]
    xmax, ymax = (
        max([point[0] for point in point_list]),
        max([point[1] for point in point_list]),
    )
    board = [[-1 for y in range(0, ymax + 1)] for x in range(-1, xmax + 1)]
    for x, y, id in point_list:
        board[x][y] = id
    return board


def transform_solution_list_to_instructions(
    solution_list: list[tuple[int, int]]
) -> str:
    instructions = []
    for index, instruction in enumerate(solution_list):
        direction = ["west", "east", "north", "south"][instruction[1]]
        instructions.append(f"{index+1}: Move fling {instruction[0]} {direction}")
    return "\n".join(instructions)


def pretty_tostr_board(
    board: list[list[int]],
    empty_icon: Optional[str] = None,
    fling_icon: Optional[str] = None,
) -> str:
    icon_0 = empty_icon if empty_icon else "x"
    icon_1 = fling_icon if fling_icon else None
    pretty_list = [""]
    for y in range(0, len(board[0])):
        for x in range(0, len(board)):
            pretty_list.append(
                icon_0 if board[x][y] == -1 else icon_1 if icon_1 else str(board[x][y])
            )
        pretty_list.append("\n")
    return " ".join(pretty_list)


def solve_fling(points: list[tuple[int, int]]) -> Union[bool, list[tuple[int, int]]]:
    processed_points = process_points_list(points)
    return recurse_fling(processed_points, [], [])


def recurse_fling(
    points: list[tuple[int, int, int]],
    solution_list: list[tuple[int, int]],
    previous_board_state: list[tuple[int, int, int]],
) -> Union[bool, list[tuple[int, int]]]:
    if previous_board_state == points:
        return False
    if len(points) == 1:
        return solution_list
    for index, point in enumerate(points):
        for direction in range(0, 4):
            local_points, local_solution_list = try_move_direction(
                *point,
                self_index=index,
                direction=direction,
                points=copy.deepcopy(points),
                solution_list=copy.deepcopy(solution_list),
            )
            next_recursion_step = recurse_fling(
                local_points, local_solution_list, points
            )
            if next_recursion_step:
                return next_recursion_step


def process_points_list(points: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    for index, point in enumerate(points):
        points[index] = (point[0], point[1], index)
    return points


def get_points_that_satisfy_condition(
    f, points: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    points_in_condition = []
    for index, point in enumerate(points):
        if f(point):
            points_in_condition.append(copy.deepcopy(points[index]))
    return points_in_condition


# this function is terrible! nested lambdas are alright I guess though
def try_move_direction(
    x: int,
    y: int,
    self_id: int,
    self_index: int,
    direction: int,
    points: list[tuple[int, int, int]],
    solution_list: list[tuple[int, int]],
    initial: bool = True,
) -> None:
    selected_points = get_points_that_satisfy_condition(
        [
            lambda point: point[0] < x and point[1] == y,
            lambda point: point[0] > x and point[1] == y,
            lambda point: point[0] == x and point[1] < y,
            lambda point: point[0] == x and point[1] > y,
        ][direction],
        points,
    )
    if len(selected_points) == 0 and not initial:
        # print(f"Removed fling {self_id} which was at {x},{y}")
        points.pop(self_index)
        return points, solution_list
    if [
        (x - 1, y, self_id),
        (x + 1, y, self_id),
        (x, y - 1, self_id),
        (x, y + 1, self_id),
    ][direction] in selected_points or len(selected_points) == 0:
        # second part should be or (len(selected_points) == 0 and initial) but above short circuits that
        return points, solution_list
    closest_fling_index = [
        lambda points: min(range(len(points)), key=lambda x: points.__getitem__(x)[0]),
        lambda points: max(range(len(points)), key=lambda x: points.__getitem__(x)[0]),
        lambda points: min(range(len(points)), key=lambda x: points.__getitem__(x)[1]),
        lambda points: max(range(len(points)), key=lambda x: points.__getitem__(x)[1]),
    ][direction](points)
    closest_point = points[closest_fling_index]
    if initial:
        solution_list.append((self_id, direction))
    points[self_index] = (
        closest_point[0] + (1 if direction == 0 else -1 if direction == 1 else 0),
        closest_point[1] + (1 if direction == 2 else -1 if direction == 3 else 0),
        self_id,
    )
    return try_move_direction(
        closest_point[0],
        closest_point[1],
        closest_point[2],
        closest_fling_index,
        direction,
        points,
        solution_list,
        False,
    )


def check_if_position_is_valid(x: int, y: int, points: list[tuple[int, int]]) -> bool:
    return not (
        (x, y) in points
        or (x + 1, y) in points
        or (x, y + 1) in points
        or (x - 1, y) in points
        or (x, y - 1) in points
    )


def test_solve_fling() -> None:
    points = [(1, 0), (1, 3), (3, 2)]
    result = solve_fling(points)
    assert result == [(0, 3), (0, 1)]


def test_create_board_from_point_list() -> None:
    points = [(1, 0), (1, 3), (3, 2)]
    result = create_board_from_point_list(points)
    assert result == [
        [0, -1, -1, 1],
        [-1, -1, -1, -1],
        [-1, -1, 2, -1],
        [-1, -1, -1, -1],
    ]


def test_get_points_that_satisfy_condition() -> None:
    points = [(1, 0, 0), (1, 3, 1), (3, 2, 2)]
    func = lambda point: point[0] == 1 and point[1] > 0
    result = get_points_that_satisfy_condition(func, points)
    assert result == [(1, 3, 1)]


def test_transform_solution_list_to_instructions() -> None:
    solution_list = [(0, 3), (0, 1)]
    result = transform_solution_list_to_instructions(solution_list)
    assert result == "1: Move fling 0 south\n2: Move fling 0 east"


def test_pretty_tostr_board() -> None:
    board = [
        [0, -1, -1, 1],
        [-1, -1, -1, -1],
        [-1, -1, 2, -1],
        [-1, -1, -1, -1],
    ]
    expected = " 0 x x x \n x x x x \n x x 2 x \n 1 x x x \n"
    result = pretty_tostr_board(board)
    assert result == expected


def dumb_board_generator(
    n: int, LIM_ITERATIONS: int = 10000, board_size: Optional[int] = None
) -> Union[list[tuple[int, int]], None]:
    iterations = 0
    while iterations < LIM_ITERATIONS:
        point_list = []
        for point in range(0, board_size if board_size else n):
            point_list.append((random.randrange(0, n + 1), random.randrange(0, n + 1)))
        if solve_fling(point_list):
            return point_list
    return []


def test_process_points_list() -> None:
    points = [(1, 0), (1, 3), (3, 2)]
    expected = [(1, 0, 0), (1, 3, 1), (3, 2, 2)]
    result = process_points_list(points)
    assert expected == result


def benchmark_solve_fling(
    max_flings: int = 10,
    iterations_for_benchmark: int = 10,
    output: bool = False,
    const_board_size: Optional[int] = None,
) -> dict[int, list[int]]:
    time_dict = {}
    for n in range(2, max_flings + 1):
        time_dict[n] = []
        time_avg = 0
        for iteration in range(iterations_for_benchmark):
            points = dumb_board_generator(n, const_board_size)
            start_time = time.time_ns()
            solve_fling(points)
            end_time = time.time_ns()
            time_taken = end_time - start_time
            time_avg += time_taken
            time_dict[n].append(time_taken)
    if output:
        print(
            "\n".join(
                [
                    f"Average Execution time for number of flings = {n}: {sum(time_ns)//iterations_for_benchmark}ns"
                    for n, time_ns in time_dict.items()
                ]
            )
        )
        graph_benchmark_times(time_dict)
    return time_dict


def graph_benchmark_times(times: dict[int, list[int]]):

    number_of_flings = [n for n, time_list in times.items() for time in time_list]
    time_values = [time for time_list in times.values() for time in time_list]

    source = pd.DataFrame(
        {"Number of flings": number_of_flings, "Time taken (ns)": time_values}
    )

    base = (
        alt.Chart(source)
        .mark_circle(opacity=0.5)
        .encode(
            alt.X("Number of flings:Q"),
            alt.Y("Time taken (ns):Q").scale(type="log"),
        )
    )

    base_no_log = (
        alt.Chart(source)
        .mark_circle(opacity=0.5)
        .encode(
            alt.X("Number of flings:Q"),
            alt.Y("Time taken (ns):Q"),
        )
    )

    base = base + base.transform_loess(
        "Number of flings", "Time taken (ns)", groupby=["category"]
    ).mark_line(size=4)

    avg_graph = base_no_log.transform_loess(
        "Number of flings", "Time taken (ns)", groupby=["category"]
    ).mark_line(size=4)
    base.save("./images/time_avg.png")
    avg_graph.save("./images/time_avg_no_log.png")
