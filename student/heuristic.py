from __future__ import annotations

import math

import numpy as np

from game2048.rules import legal_moves, move_board
from game2048.types import Move


def _snake_weights() -> list[int]:
    return [4, 3, 2, 1, 2, 3, 4, 5, 5, 4, 3, 2, 3, 2, 1, 2]


def evaluate_board(board: np.ndarray, merge_score: int = 0) -> float:
    """Heuristic chấm giá trị bàn cờ 2048 theo nhiều yếu tố cùng lúc."""
    board = np.asarray(board, dtype=np.int64)
    if board.size == 0:
        return 0.0

    largest_tile = int(board.max())
    if largest_tile == 0:
        return 0.0

    empty_cells = int(np.count_nonzero(board == 0))
    corners = [int(board[0, 0]), int(board[0, 3]), int(board[3, 0]), int(board[3, 3])]
    corner_value = max(corners)

    # 1) Không gian trống và tile lớn
    score = 220.0 * empty_cells + 32.0 * largest_tile + 12.0 * merge_score

    # 2) Ưu tiên giữ tile lớn ở góc
    score += 150.0 * corner_value
    if largest_tile in corners:
        score += 500.0

    # 3) Smoothness: board càng mượt càng tốt.
    smoothness = 0.0
    for i in range(4):
        for j in range(4):
            value = int(board[i, j])
            if value == 0:
                continue
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < 4 and 0 <= nj < 4:
                    other = int(board[ni, nj])
                    if other != 0:
                        smoothness -= abs(math.log2(value) - math.log2(other))
    score += 40.0 * smoothness

    # 4) Monotonicity: các đường hàng/cột có xu hướng sắp xếp gọn.
    monotonicity = 0.0
    for line in [board[i, :] for i in range(4)] + [board[:, j] for j in range(4)]:
        values = [int(v) for v in line if v > 0]
        if len(values) < 2:
            continue
        inc = 0
        dec = 0
        for a, b in zip(values, values[1:]):
            if b >= a:
                inc += 1
            if b <= a:
                dec += 1
        monotonicity += max(inc, dec) * 30.0
    score += monotonicity

    # 5) Merge potential: ưu tiên những ô tạo khả năng gộp tiếp.
    merge_potential = 0
    for i in range(4):
        for j in range(4):
            value = int(board[i, j])
            if value == 0:
                continue
            if i + 1 < 4 and board[i + 1, j] == value:
                merge_potential += value
            if j + 1 < 4 and board[i, j + 1] == value:
                merge_potential += value
    score += 20.0 * merge_potential

    # 6) Snake pattern: ưu tiên sắp xếp theo mô hình điềm tĩnh ở góc.
    snake = _snake_weights()
    snake_score = 0.0
    for idx, weight in enumerate(snake):
        r = idx // 4
        c = idx % 4
        snake_score += int(board[r, c]) * weight
    score += 0.7 * snake_score

    return float(score)


def choose_best_move(
    board: np.ndarray,
    legal: tuple[Move, ...],
    max_depth: int = 2,
) -> Move:
    """Chọn move tốt nhất bằng heuristic + shallow search (1 bước xem trước)."""
    if not legal:
        raise ValueError("Không có nước hợp lệ để chọn")

    best_move = legal[0]
    best_value = float("-inf")

    for move in legal:
        next_board, merge_score, changed = move_board(board, move)
        if not changed:
            continue

        value = evaluate_board(next_board, merge_score)

        if max_depth > 1:
            future_moves = legal_moves(next_board)
            if future_moves:
                next_values = []
                for next_move in future_moves:
                    future_board, future_merge_score, _ = move_board(next_board, next_move)
                    next_values.append(evaluate_board(future_board, future_merge_score))
                value += 0.35 * max(next_values)

        if value > best_value:
            best_value = value
            best_move = move

    return best_move
