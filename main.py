import src.fling as fling


def main() -> None:
    fling.test_create_board_from_point_list()
    fling.test_solve_fling()
    fling.test_get_points_that_satisfy_condition()
    fling.test_transform_solution_list_to_instructions()
    fling.test_pretty_tostr_board()
    fling.test_process_points_list()
    points = [(1, 0), (1, 3), (3, 2)]
    print(fling.pretty_tostr_board(fling.create_board_from_point_list(points)))
    result = fling.solve_fling(points)
    print(fling.transform_solution_list_to_instructions(result))
    fling.benchmark_solve_fling(10, 100, True, 7)


if __name__ == "__main__":
    main()
