"""Time-bounded expectimax search for the 2048 agent.

The search treats player moves as max nodes and random tile spawns as chance
nodes. Only fully evaluated nodes are written to the transposition table, and
iterative deepening only replaces the chosen move after a complete iteration.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from game2048.rules import legal_moves, move_board
from game2048.types import Move
from student.heuristic import evaluate_board


# board bytes, node type, depth. Empty count is derivable from board bytes.
TTKey = tuple[bytes, str, int]

# Search statistics consumed by the benchmark.
LAST_NODES = 0
LAST_COMPLETED_DEPTH = 0
_NODES_VISITED = 0


class _SearchTimeout(Exception):
    """Internal signal used to discard an incomplete search iteration."""


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


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise _SearchTimeout


def _danger_penalty(board: np.ndarray) -> float:
    """Small tie-breaking penalty for fragile board shapes.

    Keep this deliberately weaker than the main heuristic so it cannot
    dominate the evaluation or incorrectly label a board as lost.
    """
    penalty = 0.0
    corners = ((0, 0), (0, 3), (3, 0), (3, 3))

    for r, c in corners:
        if board[r, c] == 0:
            penalty += 35.0

    for r in range(4):
        for c in range(4):
            value = int(board[r, c])
            if value < 128 or (r, c) in corners:
                continue
            has_neighbor = any(
                0 <= r + dr < 4
                and 0 <= c + dc < 4
                and board[r + dr, c + dc] != 0
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            if not has_neighbor:
                penalty += min(value * 0.18, 500.0)

    return penalty


def _ordered_moves(
    board: np.ndarray,
    heuristic_fn: Callable[[np.ndarray, int], float],
    deadline: float | None = None,
) -> list[tuple[float, Move, np.ndarray, int]]:
    """Order legal moves best-first to improve early useful search results."""
    moves: list[tuple[float, Move, np.ndarray, int]] = []
    for move in legal_moves(board):
        if deadline is not None:
            _check_deadline(deadline)
        next_board, merge_score, changed = move_board(board, move)
        if not changed:
            continue
        score = float(heuristic_fn(next_board, merge_score))
        score -= 0.12 * _danger_penalty(next_board)
        moves.append((score, move, next_board, merge_score))
    moves.sort(key=lambda item: item[0], reverse=True)
    return moves


def _player_value(
    board: np.ndarray,
    depth: int,
    heuristic_fn: Callable[[np.ndarray, int], float],
    deadline: float,
    tt: dict[TTKey, float],
) -> float:
    global _NODES_VISITED
    _check_deadline(deadline)
    _NODES_VISITED += 1
    key: TTKey = (board.tobytes(), "player", depth)
    cached = tt.get(key)
    if cached is not None:
        return cached

    moves = legal_moves(board)
    if not moves:
        # A terminal board should be much worse than any playable board.
        return float(heuristic_fn(board, 0)) - 1_000_000.0
    if depth <= 0:
        return float(heuristic_fn(board, 0))

    best = float("-inf")
    # No arbitrary top-k pruning: there are at most four moves, and omitting
    # a legal move can make the agent miss the best continuation.
    for _, _, next_board, _ in _ordered_moves(board, heuristic_fn, deadline):
        _check_deadline(deadline)
        value = _chance_value(next_board, depth - 1, heuristic_fn, deadline, tt)
        if value > best:
            best = value

    # Store only after every legal move was completed successfully.
    tt[key] = best
    return best


def _chance_value(
    board: np.ndarray,
    depth: int,
    heuristic_fn: Callable[[np.ndarray, int], float],
    deadline: float,
    tt: dict[TTKey, float],
) -> float:
    """Expected value over the random 2/4 tile spawn after a player move."""
    global _NODES_VISITED
    _check_deadline(deadline)
    _NODES_VISITED += 1
    key: TTKey = (board.tobytes(), "chance", depth)
    cached = tt.get(key)
    if cached is not None:
        return cached

    empties = np.argwhere(board == 0)
    if depth <= 0 or len(empties) == 0:
        value = float(heuristic_fn(board, 0))
        tt[key] = value
        return value

    # Each empty position is equally likely; the spawned tile is 2 with 0.9
    # probability and 4 with 0.1 probability.
    position_probability = 1.0 / len(empties)
    expected = 0.0
    for r, c in empties:
        _check_deadline(deadline)
        for tile, tile_probability in ((2, 0.9), (4, 0.1)):
            spawned = board.copy()
            spawned[int(r), int(c)] = tile
            expected += position_probability * tile_probability * _player_value(
                spawned, depth, heuristic_fn, deadline, tt
            )

    # Store only the fully completed expectation.
    tt[key] = expected
    return expected


def shallow_search_best_move(
    board: np.ndarray,
    legal: tuple[Move, ...],
    depth: int = 3,
    heuristic_fn: Callable[[np.ndarray, int], float] = evaluate_board,
    time_limit_ms: float = 12.0,
) -> Move:
    """Return a legal move using time-bounded iterative-deepening expectimax.

    A heuristic one-ply move is prepared first, so a very small time budget
    still produces a valid answer. A deeper result is accepted only when the
    entire iteration completes before the deadline.
    """
    if not legal:
        raise ValueError("Không có move hợp lệ để xét")

    global LAST_NODES, LAST_COMPLETED_DEPTH, _NODES_VISITED
    LAST_NODES = 0
    LAST_COMPLETED_DEPTH = 0
    _NODES_VISITED = 0

    start = time.perf_counter()
    # Leave a small margin for returning through the agent/API wrapper.
    budget_seconds = max(0.001, float(time_limit_ms) / 1000.0 * 0.92)
    deadline = start + budget_seconds

    try:
        ordered = _ordered_moves(board, heuristic_fn)
    except _SearchTimeout:
        return legal[0]

    # Respect the supplied legal-move list and keep a safe fallback.
    legal_set = set(legal)
    ordered = [item for item in ordered if item[1] in legal_set]
    if not ordered:
        return legal[0]

    best_move = ordered[0][1]
    phase = _phase(board)
    if phase == "early":
        max_depth = min(5, max(1, int(depth)))
    elif phase == "mid":
        max_depth = min(6, max(2, int(depth)))
    else:
        max_depth = min(7, max(2, int(depth)))

    tt: dict[TTKey, float] = {}
    for current_depth in range(1, max_depth + 1):
        local_best_move = best_move
        local_best_value = float("-inf")
        try:
            for _, move, next_board, _ in ordered:
                _check_deadline(deadline)
                value = _chance_value(
                    next_board, current_depth - 1, heuristic_fn, deadline, tt
                )
                # A small immediate preference helps resolve shallow ties but
                # does not repeatedly add the same heuristic at every level.
                immediate = float(heuristic_fn(next_board, 0))
                value = 0.85 * value + 0.15 * immediate
                if value > local_best_value:
                    local_best_value = value
                    local_best_move = move
        except _SearchTimeout:
            # Discard the incomplete iteration; keep the last complete result.
            break

        best_move = local_best_move
        LAST_COMPLETED_DEPTH = current_depth
        LAST_NODES = _NODES_VISITED

    return best_move
