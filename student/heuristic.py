from __future__ import annotations

import math

import numpy as np

from game2048.rules import legal_moves, move_board
from game2048.types import Move


BASE_WEIGHTS = {
    "empty": 520.0,
    "largest_tile": 42.0,
    "merge_score": 12.0,
    "corner_value": 260.0,
    "corner_bonus": 3200.0,
    "corner_penalty": 2200.0,
    "smoothness": 24.0,
    "monotonicity": 70.0,
    "merge_potential": 36.0,
    "snake": 17.0,
    "isolated_penalty": 1.5,
    "empty_corner": 15.0,
    "corner_stability": 18.0,
    "dead_board": 8.0,
    "hole_penalty": 12.0,
}


def _phase(board: np.ndarray) -> str:
    empty_cells = int(np.count_nonzero(board == 0))
    max_tile = int(board.max())
    if max_tile >= 2048 or empty_cells <= 4:
        return "late"
    if max_tile >= 512 or empty_cells <= 9:
        return "mid"
    return "early"


def _phase_weights(board: np.ndarray) -> dict[str, float]:
    phase = _phase(board)
    weights = BASE_WEIGHTS.copy()

    if phase == "early":
        weights.update(
            {
                "empty": 660.0,
                "largest_tile": 20.0,
                "merge_score": 18.0,
                "corner_value": 220.0,
                "corner_bonus": 2600.0,
                "corner_penalty": 2400.0,
                "smoothness": 18.0,
                "monotonicity": 48.0,
                "merge_potential": 52.0,
                "snake": 20.0,
                "isolated_penalty": 1.8,
                "empty_corner": 18.0,
                "corner_stability": 24.0,
                "dead_board": 10.0,
                "hole_penalty": 14.0,
            }
        )
    elif phase == "mid":
        weights.update(
            {
                "empty": 560.0,
                "largest_tile": 28.0,
                "merge_score": 16.0,
                "corner_value": 260.0,
                "corner_bonus": 3000.0,
                "corner_penalty": 2300.0,
                "smoothness": 30.0,
                "monotonicity": 75.0,
                "merge_potential": 46.0,
                "snake": 18.0,
                "isolated_penalty": 2.2,
                "empty_corner": 16.0,
                "corner_stability": 22.0,
                "dead_board": 9.0,
                "hole_penalty": 13.0,
            }
        )
    else:
        weights.update(
            {
                "empty": 420.0,
                "largest_tile": 48.0,
                "merge_score": 14.0,
                "corner_value": 320.0,
                "corner_bonus": 4200.0,
                "corner_penalty": 2600.0,
                "smoothness": 36.0,
                "monotonicity": 90.0,
                "merge_potential": 58.0,
                "snake": 15.0,
                "isolated_penalty": 2.8,
                "empty_corner": 12.0,
                "corner_stability": 28.0,
                "dead_board": 12.0,
                "hole_penalty": 18.0,
            }
        )
    return weights


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
            bonus -= 80.0
    for r, c in (
        (0, 1), (1, 0), (0, 2), (2, 0),
        (1, 1), (2, 2), (3, 2), (2, 3),
        (1, 3), (3, 1),
    ):
        if board[r, c] == 0:
            bonus += 8.0
    return bonus


def _line_smoothness(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    score = 0.0
    for a, b in zip(values, values[1:]):
        if a == 0 or b == 0:
            continue
        score -= abs(math.log2(a) - math.log2(b)) * min(a, b)
    return score


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


def _real_merge_potential(board: np.ndarray) -> float:
    potential = 0.0
    lines = []
    for row in board:
        lines.append([int(v) for v in row if v != 0])
    for col in board.T:
        lines.append([int(v) for v in col if v != 0])

    for values in lines:
        if len(values) < 2:
            continue
        for i in range(len(values) - 1):
            a, b = values[i], values[i + 1]
            if a == b:
                potential += a * 2.0
            elif abs(math.log2(a) - math.log2(b)) <= 1.0:
                potential += min(a, b) * 0.8
        for i in range(len(values) - 2):
            if values[i] == values[i + 2]:
                potential += values[i] * 0.6
    return potential + 0.5 * _merge_chain_potential(board)


def _corner_stability(board: np.ndarray) -> float:
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    corner_values = [int(board[r, c]) for r, c in corners]
    largest_corner = max(corner_values)
    score = 0.0

    for r, c in corners:
        value = int(board[r, c])
        if value == 0:
            score -= 120.0
        else:
            score += value * 0.22
            if value >= 512:
                score += 40.0

    if largest_corner > 0:
        score += 0.35 * largest_corner

    for r, c in corners:
        if int(board[r, c]) == int(board.max()) and int(board.max()) >= 128:
            score += 35.0
            for rr, cc in ((r, 1 if c == 0 else 2), (1 if r == 0 else 2, c)):
                if 0 <= rr < 4 and 0 <= cc < 4:
                    score += 0.1 * int(board[rr, cc])

    for r, c in corners:
        if board[r, c] == 0:
            if int(board.max()) >= 128:
                score -= 80.0
    return score


def _dead_board_penalty(board: np.ndarray) -> float:
    penalty = 0.0
    for r in range(4):
        for c in range(4):
            value = int(board[r, c])
            if value < 128 or (r, c) in ((0, 0), (0, 3), (3, 0), (3, 3)):
                continue
            support = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 4 and 0 <= cc < 4 and board[rr, cc] != 0:
                    if int(board[rr, cc]) >= value / 2:
                        support += 1
            if support == 0:
                penalty += value
    return penalty


def _hole_penalty(board: np.ndarray) -> float:
    penalty = 0.0
    for row in board:
        vals = [int(v) for v in row]
        for idx in range(len(vals) - 1):
            if vals[idx] == 0 and vals[idx + 1] != 0:
                penalty += vals[idx + 1] * 0.35
    for col in board.T:
        vals = [int(v) for v in col]
        for idx in range(len(vals) - 1):
            if vals[idx] == 0 and vals[idx + 1] != 0:
                penalty += vals[idx + 1] * 0.35
    return penalty


def _isolated_big_tile_penalty(board: np.ndarray) -> float:
    penalty = 0.0
    for r in range(4):
        for c in range(4):
            value = int(board[r, c])
            if value < 128:
                continue
            if (r, c) in ((0, 0), (0, 3), (3, 0), (3, 3)):
                continue
            neighbors = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < 4 and 0 <= cc < 4 and board[rr, cc] != 0:
                    neighbors += 1
            if neighbors == 0:
                penalty += value
    return penalty


def evaluate_board(board: np.ndarray, merge_score: int = 0) -> float:
    board = np.asarray(board, dtype=np.int64)
    if board.size == 0:
        return 0.0

    largest_tile = int(board.max())
    if largest_tile == 0:
        return 0.0

    weights = _phase_weights(board)
    empty_cells = int(np.count_nonzero(board == 0))
    corners = [int(board[0, 0]), int(board[0, 3]), int(board[3, 0]), int(board[3, 3])]
    corner_value = max(corners)
    corner_matrix = _corner_matrix()

    score = 0.0
    score += float(np.sum(board * corner_matrix)) * 0.12
    score += weights["empty"] * empty_cells
    score += weights["largest_tile"] * largest_tile
    score += weights["merge_score"] * merge_score
    score += weights["corner_value"] * corner_value
    score += weights["corner_stability"] * _corner_stability(board)
    if largest_tile in corners:
        score += weights["corner_bonus"]
    else:
        score -= weights["corner_penalty"]
    score += _empty_corner_bonus(board) * weights["empty_corner"]

    smoothness = 0.0
    for r in range(4):
        row = [int(v) for v in board[r, :] if v > 0]
        smoothness += _line_smoothness(row)
    for c in range(4):
        col = [int(v) for v in board[:, c] if v > 0]
        smoothness += _line_smoothness(col)
    score += weights["smoothness"] * smoothness

    monotonicity = 0.0
    for row in board:
        values = [int(v) for v in row if v > 0]
        monotonicity += _line_monotonicity(values)
    for col in board.T:
        values = [int(v) for v in col if v > 0]
        monotonicity += _line_monotonicity(values)
    score += weights["monotonicity"] * monotonicity

    merge_potential = _real_merge_potential(board)
    score += weights["merge_potential"] * merge_potential

    snake_score = 0.0
    snake_weights = _snake_weights()
    for idx, weight in enumerate(snake_weights):
        r = idx // 4
        c = idx % 4
        snake_score += int(board[r, c]) * weight
    score += weights["snake"] * snake_score

    isolated_penalty = _isolated_big_tile_penalty(board)
    score -= weights["isolated_penalty"] * isolated_penalty
    score -= weights["dead_board"] * _dead_board_penalty(board)
    score -= weights["hole_penalty"] * _hole_penalty(board)
    return float(score)


def choose_best_move(board: np.ndarray, legal: tuple[Move, ...], max_depth: int = 2) -> Move:
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
