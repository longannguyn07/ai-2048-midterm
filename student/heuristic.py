from __future__ import annotations

import math

import numpy as np

from game2048.rules import legal_moves, move_board
from game2048.types import Move


TUNED_WEIGHTS = {
    "empty": 650.0,
    "largest_tile": 26.0,
    "merge_score": 11.5,
    "corner_value": 240.0,
    "corner_bonus": 2800.0,
    "corner_penalty": 1900.0,
    "smoothness": 20.5,
    "monotonicity": 54.0,
    "merge_potential": 32.0,
    "chain_potential": 18.5,
    "snake": 14.0,
    "isolated_penalty": 0.9,
    "empty_corner": 12.0,
}


def _snake_weights() -> list[int]:
    return [
        4, 3, 2, 1,
        3, 2, 1, 2,
        2, 1, 2, 3,
        1, 2, 3, 4,
    ]


def _corner_matrix() -> np.ndarray:
    return np.array(
        [
            [65536, 32768, 16384, 8192],
            [32768, 16384, 8192, 4096],
            [16384, 8192, 4096, 2048],
            [8192, 4096, 2048, 1024],
        ],
        dtype=np.float64,
    )


def _line_monotonicity(values: list[int]) -> int:
    if len(values) < 2:
        return 0
    inc = sum(1 for a, b in zip(values, values[1:]) if b >= a)
    dec = sum(1 for a, b in zip(values, values[1:]) if b <= a)
    return max(inc, dec)


def _empty_corner_bonus(board: np.ndarray) -> float:
    bonus = 0.0
    for r, c in ((0, 0), (0, 3), (3, 0), (3, 3)):
        if board[r, c] == 0:
            bonus -= 60.0
    for r, c in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 1), (2, 2), (3, 2), (2, 3), (1, 3), (3, 1)):
        if board[r, c] == 0:
            bonus += 8.0
    return bonus * TUNED_WEIGHTS["empty_corner"]


def _merge_chain_potential(board: np.ndarray) -> float:
    potential = 0.0
    for r in range(4):
        for c in range(4):
            value = int(board[r, c])
            if value == 0:
                continue
            chain = 0
            for dc in (1, -1):
                rr, cc = r, c
                while 0 <= rr < 4 and 0 <= cc < 4:
                    cc += dc
                    if 0 <= cc < 4 and board[rr, cc] == value:
                        chain += 1
                    else:
                        break
            for dr in (1, -1):
                rr, cc = r, c
                while 0 <= rr < 4 and 0 <= cc < 4:
                    rr += dr
                    if 0 <= rr < 4 and board[rr, cc] == value:
                        chain += 1
                    else:
                        break
            if chain > 0:
                potential += value * (chain / 4.0)
    return potential


def evaluate_board(board: np.ndarray, merge_score: int = 0) -> float:
    board = np.asarray(board, dtype=np.int64)
    if board.size == 0:
        return 0.0

    largest_tile = int(board.max())
    if largest_tile == 0:
        return 0.0

    empty_cells = int(np.count_nonzero(board == 0))
    corners = [int(board[0, 0]), int(board[0, 3]), int(board[3, 0]), int(board[3, 3])]
    corner_value = max(corners)
    corner_matrix = _corner_matrix()

    score = 0.0
    score += float(np.sum(board * corner_matrix)) * 0.12
    score += TUNED_WEIGHTS["empty"] * empty_cells + TUNED_WEIGHTS["largest_tile"] * largest_tile + TUNED_WEIGHTS["merge_score"] * merge_score
    score += TUNED_WEIGHTS["corner_value"] * corner_value
    if largest_tile in corners:
        score += TUNED_WEIGHTS["corner_bonus"]
    else:
        score -= TUNED_WEIGHTS["corner_penalty"]
    score += _empty_corner_bonus(board)

    smoothness = 0.0
    for r in range(4):
        for c in range(4):
            value = int(board[r, c])
            if value == 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < 4 and 0 <= cc < 4:
                    other = int(board[rr, cc])
                    if other != 0:
                        smoothness -= abs(math.log2(value) - math.log2(other)) * min(value, other)
    score += TUNED_WEIGHTS["smoothness"] * smoothness

    monotonicity = 0.0
    for row in board:
        values = [int(v) for v in row if v > 0]
        monotonicity += _line_monotonicity(values)
    for col in board.T:
        values = [int(v) for v in col if v > 0]
        monotonicity += _line_monotonicity(values)
    score += TUNED_WEIGHTS["monotonicity"] * monotonicity

    merge_potential = 0
    for r in range(4):
        for c in range(4):
            value = int(board[r, c])
            if value == 0:
                continue
            if r + 1 < 4 and board[r + 1, c] == value:
                merge_potential += value
            if c + 1 < 4 and board[r, c + 1] == value:
                merge_potential += value
    score += TUNED_WEIGHTS["merge_potential"] * merge_potential + TUNED_WEIGHTS["chain_potential"] * _merge_chain_potential(board)

    snake = _snake_weights()
    snake_score = 0.0
    for idx, weight in enumerate(snake):
        r = idx // 4
        c = idx % 4
        snake_score += int(board[r, c]) * weight
    score += TUNED_WEIGHTS["snake"] * snake_score

    isolated_penalty = 0
    for r in range(4):
        for c in range(4):
            value = int(board[r, c])
            if value < 64:
                continue
            neighbors = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < 4 and 0 <= cc < 4 and board[rr, cc] != 0:
                    neighbors += 1
            if neighbors == 0:
                isolated_penalty += value
    score -= TUNED_WEIGHTS["isolated_penalty"] * isolated_penalty
    return float(score)


def choose_best_move(
    board: np.ndarray,
    legal: tuple[Move, ...],
    max_depth: int = 2,
) -> Move:
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
