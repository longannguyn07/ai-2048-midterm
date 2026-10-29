from __future__ import annotations

import time
from typing import Callable

import numpy as np

from game2048.rules import legal_moves, move_board
from game2048.types import Move
from student.heuristic import evaluate_board


TTKey = tuple[bytes, str, int, int]


def _empty_cells(board: np.ndarray) -> int:
    return int(np.count_nonzero(board == 0))


def _phase(board: np.ndarray) -> str:
    empty = _empty_cells(board)
    max_tile = int(board.max())
    if max_tile >= 1024 or empty <= 5:
        return "late"
    if max_tile >= 512 or empty <= 10:
        return "mid"
    return "early"


def _safe_move_bonus(board: np.ndarray, next_board: np.ndarray, merge_score: int) -> float:
    empty_gain = _empty_cells(next_board) - _empty_cells(board)
    score = 30.0 * empty_gain + 12.0 * merge_score

    corners = ((0, 0), (0, 3), (3, 0), (3, 3))
    corner_value = max(int(next_board[r, c]) for r, c in corners)
    score += 40.0 * corner_value

    danger = 0.0
    for r, c in corners:
        if next_board[r, c] == 0:
            danger += 80.0
    score -= danger

    big_tiles = 0
    for r in range(4):
        for c in range(4):
            value = int(next_board[r, c])
            if value >= 128:
                big_tiles += 1
    score -= 10.0 * max(0, big_tiles - 4)
    return score


def _ordered_moves(board: np.ndarray, heuristic_fn: Callable[[np.ndarray, int], float]) -> list[tuple[float, Move, np.ndarray, int]]:
    moves: list[tuple[float, Move, np.ndarray, int]] = []
    for move in legal_moves(board):
        next_board, merge_score, changed = move_board(board, move)
        if not changed:
            continue
        immediate = heuristic_fn(next_board, merge_score)
        safe_bonus = _safe_move_bonus(board, next_board, merge_score)
        score = immediate + 0.18 * safe_bonus
        moves.append((score, move, next_board, merge_score))
    moves.sort(key=lambda item: item[0], reverse=True)
    return moves


def _endgame_exactish(board: np.ndarray, heuristic_fn: Callable[[np.ndarray, int], float]) -> float:
    empty = _empty_cells(board)
    if empty > 6:
        return heuristic_fn(board, 0)

    best = float("-inf")
    for move in legal_moves(board):
        next_board, merge_score, changed = move_board(board, move)
        if not changed:
            continue
        value = heuristic_fn(next_board, merge_score)
        if value > best:
            best = value
    return best


def _search_chance(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    heuristic_fn: Callable[[np.ndarray, int], float],
    start_time: float,
    time_limit_ms: float,
    tt: dict[TTKey, float],
) -> float:
    if time.time() - start_time > time_limit_ms / 1000.0:
        return float(heuristic_fn(board, 0))

    key = (board.tobytes(), "chance", depth, _empty_cells(board))
    if key in tt:
        return tt[key]

    if depth <= 0 or not legal_moves(board):
        value = float(heuristic_fn(board, 0))
        tt[key] = value
        return value

    empties = np.argwhere(board == 0)
    if len(empties) == 0:
        value = float(heuristic_fn(board, 0))
        tt[key] = value
        return value

    total = 0.0
    for r, c in empties:
        for tile, weight in ((2, 0.9), (4, 0.1)):
            spawn = board.copy()
            spawn[r, c] = tile
            total += weight * _search_player(spawn, depth - 1, alpha, beta, heuristic_fn, start_time, time_limit_ms, tt)
    expected = total / len(empties)
    tt[key] = expected
    return expected


def _search_player(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    heuristic_fn: Callable[[np.ndarray, int], float],
    start_time: float,
    time_limit_ms: float,
    tt: dict[TTKey, float],
) -> float:
    if time.time() - start_time > time_limit_ms / 1000.0:
        return float(heuristic_fn(board, 0))

    key = (board.tobytes(), "player", depth, _empty_cells(board))
    if key in tt:
        return tt[key]

    moves = legal_moves(board)
    if not moves:
        tt[key] = float("-inf")
        return float("-inf")

    if depth <= 0:
        value = float(_endgame_exactish(board, heuristic_fn)) if _empty_cells(board) <= 6 else float(heuristic_fn(board, 0))
        tt[key] = value
        return value

    best = float("-inf")
    for _, move, next_board, merge_score in _ordered_moves(board, heuristic_fn):
        if time.time() - start_time > time_limit_ms / 1000.0:
            break
        score = _search_chance(next_board, depth - 1, alpha, beta, heuristic_fn, start_time, time_limit_ms, tt)
        value = float(heuristic_fn(next_board, merge_score)) + 0.7 * score
        if value > best:
            best = value
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    tt[key] = best
    return best


def shallow_search_best_move(
    board: np.ndarray,
    legal: tuple[Move, ...],
    depth: int = 3,
    heuristic_fn: Callable[[np.ndarray, int], float] = evaluate_board,
    time_limit_ms: float = 12.0,
) -> Move:
    if not legal:
        raise ValueError("Không có move hợp lệ để xét")

    start = time.time()
    phase = _phase(board)
    if phase == "early":
        max_depth = min(6, max(2, depth))
    elif phase == "mid":
        max_depth = min(7, max(3, depth))
    else:
        max_depth = min(8, max(4, depth))

    ordered = _ordered_moves(board, heuristic_fn)
    if not ordered:
        return legal[0]

    best_move = ordered[0][1]
    best_value = float("-inf")

    for current_depth in range(1, max_depth + 1):
        if time.time() - start > time_limit_ms / 1000.0:
            break

        local_tt: dict[TTKey, float] = {}
        local_best_move = ordered[0][1]
        local_best_value = float("-inf")

        for _, move, next_board, merge_score in ordered:
            if time.time() - start > time_limit_ms / 1000.0:
                break

            immediate = heuristic_fn(next_board, merge_score)
            safe_bonus = _safe_move_bonus(board, next_board, merge_score)
            evaluated = immediate + 0.2 * safe_bonus

            if _empty_cells(board) <= 6:
                chance_value = _endgame_exactish(next_board, heuristic_fn)
            else:
                chance_value = _search_chance(next_board, current_depth - 1, float("-inf"), float("inf"), heuristic_fn, start, time_limit_ms, local_tt)

            total = evaluated + 0.8 * chance_value
            if total > local_best_value:
                local_best_value = total
                local_best_move = move

        if local_best_value > best_value:
            best_value = local_best_value
            best_move = local_best_move

    return best_move
